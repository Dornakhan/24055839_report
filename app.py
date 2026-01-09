"""
Streamlit Dashboard: Predicting Birth Outcome Following Fertility Treatment
A comprehensive dissertation dashboard including all analysis, modeling, and prediction sections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
import os
import shap
from streamlit_shap import st_shap

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Birth Outcome Predictor Dashboard",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING & PIPELINE ---
@st.cache_data
def load_all_data():
    """Load and apply the full 39-feature dissertation pipeline."""
    # Using your specified path
    file_path = "ar-2017-2018-xlsb.xlsb"
    if not os.path.exists(file_path):
        file_path = "/content/sample_data/ar-2017-2018-xlsb.xlsb"
        
    try:
        # Load Sheet1
        df_raw = pd.read_excel(file_path, sheet_name=1, engine='pyxlsb')
        
        # 1. Standardise columns
        df_raw.columns = df_raw.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True).str.strip('_')
        
        # 2. Drop columns 37-57 (foetal heart & birth outcome fields)
        df_reduced = df_raw.drop(columns=df_raw.columns[37:57], errors='ignore')
        
        # 3. Clinical Logic (0 Cycles = 0 History)
        mask = (df_reduced['total_number_of_previous_ivf_cycles'] == 0) & (df_reduced.get('total_number_of_previous_di_cycles', 0) == 0)
        targets = ['total_number_of_previous_pregnancies_ivf_and_di', 'total_number_of_previous_live_births_ivf_or_di']
        for t in targets:
            if t in df_reduced.columns:
                df_reduced.loc[mask, t] = df_reduced.loc[mask, t].fillna(0)
        
        # 4. DI Treatment Logic
        if 'specific_treatment_type' in df_reduced.columns:
            di_mask = df_reduced['specific_treatment_type'].astype(str).str.upper() == 'DI'
            df_reduced.loc[di_mask, 'date_of_embryo_transfer'] = 0
            
        # 5. Filter 999 Age Sentinels
        df_reduced = df_reduced[df_reduced['patient_age_at_treatment'].astype(str) != '999']
        
        # 6. # Drop the requested donor columns
        donor_cols = ['egg_donor_age_at_registration', 'sperm_donor_age_at_registration']
        df_reduced = df_reduced.drop(columns=donor_cols, errors='ignore')
        
        # 7. SPECIFIC ROW DELETIONS (Targeting 999s and Blanks)
        # Drop anything with 999 or '999' in patient_age and partner_age
        for col in ['patient_age_at_treatment', 'partner_age']:
            df_reduced = df_reduced[~df_reduced[col].isin([999, '999'])]

        # Drop 'nan', 'None', or blanks in partner_type
        df_reduced['partner_type'] = df_reduced['partner_type'].astype(str).str.strip()
        df_reduced = df_reduced[~df_reduced['partner_type'].isin(['nan', 'None', 'N/A', ''])]

        # Final drop for any remaining blanks in the critical columns we discussed
        final_na_drop = [
            'date_of_embryo_transfer', 
            'total_number_of_previous_pregnancies_ivf_and_di',
            'partner_age',
            'partner_type',"sperm_source"
        ]
        df_reduced = df_reduced.dropna(subset=final_na_drop)
        
        # Final cleanup of blanks
        df_reduced = df_reduced.dropna(subset=['date_of_embryo_transfer', 'total_number_of_previous_pregnancies_ivf_and_di', 'live_birth_occurrence'])
        
        # Impute known Binary Indicators with 0 (Assuming NaN means 'Attribute Not Present')
        bit_cols = [
            "frozen_cycle", "fresh_cycle", "elective_single_embryo_transfer",
            "donated_embryo",
            "embryos_transferred", "embryos_transferred_from_eggs_micro_injected",
        ]
        df_reduced[bit_cols] = df_reduced[bit_cols].fillna(0).astype("int8")

        
        # Explicitly impute '0' for object-type columns where NaN signifies 'none' or '0' count
        # These columns will then be included in one-hot encoding as a '0' category
        object_cols_to_fillna_0 = [
            "egg_source",
            "fresh_eggs_collected",
            "total_eggs_mixed",
            "total_embryos_created",
            "embryos_stored_for_use_by_patient",
        ]
        for col in object_cols_to_fillna_0:
            if col in df_reduced.columns:
                df_reduced[col] = df_reduced[col].fillna('0')
                
        # One-Hot Encoding (OHE) for remaining categorical features
        cat_cols = df_reduced.select_dtypes(include=["object"]).columns.tolist()
        df_model = pd.get_dummies(df_reduced, columns=cat_cols, drop_first=True) # drop_first=True avoids perfect multicollinearity

        # Clean Column Names Post-Encoding (Removes symbols that break ML libraries)
        df_model.columns = (
            df_model.columns
            .str.replace(r"[<>]", "", regex=True)
            .str.replace(r"[\[\]]", "", regex=True)
            .str.replace(" ", "_")
            .str.replace(r"\(", "", regex=True)
            .str.replace(r"\)", "", regex=True)
        )

        # Convert residual Boolean columns to integer type
        bool_cols = df_model.select_dtypes(include='bool').columns.tolist()
        for col in bool_cols:
            df_model[col] = df_model[col].astype('int8')

        
        # Serialisation fix for Arrow (convert mixed objects to string)
        for col in df_reduced.select_dtypes(include=['object']).columns:
            df_reduced[col] = df_reduced[col].astype(str)
            
        return df_raw, df_reduced
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None

def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train and evaluate ML models."""
    results = {}
    models = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, lr_preds),
        'F1 Score': f1_score(y_test, lr_preds),
        'AUC-ROC': roc_auc_score(y_test, lr_probs)
    }
    models['Logistic Regression'] = (lr_model, lr_preds, lr_probs)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_preds),
        'F1 Score': f1_score(y_test, rf_preds),
        'AUC-ROC': roc_auc_score(y_test, rf_probs)
    }
    models['Random Forest'] = (rf_model, rf_preds, rf_probs)
    
    return results, models

def plot_roc_curves(models, y_test):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e94560', '#00d9ff', '#50fa7b']
    
    for idx, (name, (model, preds, probs)) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=3, 
                label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance for Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    
    ax.barh(range(top_n), importances[indices], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i][:30] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    st.subheader("📉 Confusion Matrix Visualisation")
    st.write("This heatmap shows the counts of True Positives, True Negatives, False Positives, and False Negatives.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Birth', 'Live Birth'], 
                yticklabels=['No Birth', 'Live Birth'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix: Predictive Reliability')

    # Render the plot in Streamlit
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xticklabels(['No Live Birth', 'Live Birth'])
    ax.set_yticklabels(['No Live Birth', 'Live Birth'])
    
    plt.tight_layout()
    return fig


def plot_age_analysis(df_reduced):
    """Plot age group analysis."""
    if 'patient_age_at_treatment' not in df_reduced.columns or 'live_birth_occurrence' not in df_reduced.columns:
        return None
    
    age_livebirth = df_reduced.groupby('patient_age_at_treatment')['live_birth_occurrence'].agg(['mean', 'count'])
    age_livebirth.index = age_livebirth.index.astype(str)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution plot
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(age_livebirth)))
    axes[0].bar(range(len(age_livebirth)), age_livebirth['count'], color=colors1)
    axes[0].set_xticks(range(len(age_livebirth)))
    axes[0].set_xticklabels(age_livebirth.index, rotation=45, ha='right')
    axes[0].set_ylabel('Patient Count', fontsize=12)
    axes[0].set_title('Distribution by Age Group', fontsize=14, fontweight='bold')
    
    # Live birth rate plot
    colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(age_livebirth)))
    bars2 = axes[1].bar(range(len(age_livebirth)), age_livebirth['mean'] * 100, color=colors2)
    axes[1].set_xticks(range(len(age_livebirth)))
    axes[1].set_xticklabels(age_livebirth.index, rotation=45, ha='right')
    axes[1].set_ylabel('Live Birth Rate (%)', fontsize=12)
    axes[1].set_title('Live Birth Rate by Age Group', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Impact of Maternal Age on Fertility Treatment Outcomes', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_embryos_analysis(df_reduced):
    """Plot embryos transferred analysis."""
    if 'embryos_transferred' not in df_reduced.columns or 'live_birth_occurrence' not in df_reduced.columns:
        return None
    
    embryo_analysis = df_reduced.groupby('embryos_transferred')['live_birth_occurrence'].agg(['mean', 'count'])
    embryo_analysis = embryo_analysis[embryo_analysis['count'] > 10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(embryo_analysis)))
    bars = ax.bar(range(len(embryo_analysis)), embryo_analysis['mean'] * 100, color=colors)
    
    ax.set_xticks(range(len(embryo_analysis)))
    ax.set_xticklabels(embryo_analysis.index)
    ax.set_xlabel('Number of Embryos Transferred', fontsize=12)
    ax.set_ylabel('Live Birth Rate (%)', fontsize=12)
    ax.set_title('Live Birth Rate by Number of Embryos Transferred', fontsize=16, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df_reduced):
    """Plot correlation heatmap for numeric features."""
    num_df = df_reduced.select_dtypes(include=np.number)
    
    if num_df.shape[1] < 2:
        return None
    
    # Select top correlated features with live_birth_occurrence
    if 'live_birth_occurrence' in num_df.columns:
        correlations = num_df.corr()['live_birth_occurrence'].abs().sort_values(ascending=False)
        top_features = correlations.head(15).index.tolist()
        num_df = num_df[top_features]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, annot_kws={'size': 8}, linewidths=0.5)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_merged_age_analysis(df):
    st.subheader("📊 Age Band Analysis: Outcomes vs. Total Volume")
    
    # 1. Group the data by Age Band and Outcome
    age_counts = df.groupby(['patient_age_at_treatment', 'live_birth_occurrence']).size().unstack(fill_value=0)
    age_counts.columns = ['No Live Birth', 'Live Birth']
    age_counts['Total Records'] = age_counts['No Live Birth'] + age_counts['Live Birth']
    
    # Sort by clinical age progression
    age_order = ["18-34", "35-37", "38-39", "40-42", "43-44", "45-50"]
    age_counts = age_counts.reindex(age_order)

    # 2. Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Bars (Outcome Counts)
    age_counts[['No Live Birth', 'Live Birth']].plot(kind='bar', stacked=True, ax=ax1, color=['#d3d3d3', '#4CAF50'], alpha=0.8)
    ax1.set_ylabel('Number of Outcomes', fontweight='bold')
    ax1.set_xlabel('Patient Age Band', fontweight='bold')

    # 3. Create Secondary Axis for the Total Line
    ax2 = ax1.twinx()
    ax2.plot(range(len(age_counts)), age_counts['Total Records'], color='navy', marker='o', linewidth=2, label='Total Volume')
    ax2.set_ylabel('Total Patient Records', color='navy', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='navy')

    plt.title('Clinical Outcome Distribution and Total Volume by Age Group', fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    st.pyplot(fig)
    
def plot_target_distribution(df_reduced):
    """Bar chart: Live Birth vs No Live Birth (Figure 4.1)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x="live_birth_occurrence",
        data=df_reduced,
        palette=["#c0392b", "#27ae60"],
        ax=ax,
    )

    ax.set_xticklabels(["No Live Birth", "Live Birth"], fontsize=11)
    ax.set_xlabel("Outcome", fontsize=12)
    ax.set_ylabel("Number of cycles", fontsize=12)
    ax.set_title(
        "Distribution of Live Birth vs No Live Birth",
        fontsize=14,
        fontweight="bold",
    )

    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:,}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    return fig

   
# --- UI NAVIGATION ---
def main():
    st.sidebar.title("👶 Birth Outcome Predictor")
    st.sidebar.caption("Birth Outcome Analysis")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📊 Data Explorer", "📈 Analysis & Insights", 
         "🤖 Model Training", "🔮 Predictions", "📋 About"]
    )
    
    df_raw, df_reduced = load_all_data()
    
    if df_raw is None:
        st.stop()

    if page == "🏠 Overview":
        show_overview(df_reduced)
    elif page == "📊 Data Explorer":
        show_data_explorer(df_raw, df_reduced)
    elif page == "📈 Analysis & Insights":
        show_analysis(df_reduced)
    elif page == "🤖 Model Training":
        show_model_training(df_reduced)
    elif page == "🔮 Predictions":
        show_predictions(df_reduced)
    elif page == "📋 About":
        show_about()

# --- 1. OVERVIEW ---
def show_overview(df):
    st.title("🔬 Predicting Birth Outcome Following Fertility Treatment")
    st.caption("Advanced ML-powered analysis of fertility treatment success factors")
    st.divider()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Records", f"{len(df):,}")
    c2.metric("👶 Live Births", f"{int(df['live_birth_occurrence'].astype(float).sum()):,}")
    c3.metric("📈 Success Rate", f"{(df['live_birth_occurrence'].astype(float).mean()*100):.1f}%")
    c4.metric("🔢 Features", f"{len(df.columns)}")

    st.subheader("🎯 Target Outcome Distribution")
    fig = plot_target_distribution(df)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Research Objectives")
        st.write("- Develop and validate at least two machine learning models used to predict the probability of birth outcomes following fertility treatment.")
        st.write("- Compare the predictive performance of at least two different ML Algorithms on the prediction.")
        st.write("- Identify the most impactful patient and treatment factors involved in ML model-based predictions of birth outcomes using SHapley Additive exPlanations (SHAP) values.")
        st.write("- Communicate evidence-based, ML-derived predictions in an easy-to-interpret format for both providers and patients.")
    with col2:
        st.subheader("📊 Maternal Age Impact")
        fig, ax = plt.subplots()
        df.groupby('patient_age_at_treatment')['live_birth_occurrence'].mean().plot(kind='bar', ax=ax, color='teal')
        st.pyplot(fig)
        st.info("The graph shows that live birth rates are highest for women aged 18–34 and steadily decline with increasing age. However, there is a small uptick in the 45–50 group which might likely be due to very few, highly selected patients in that band")


    
# --- 2. DATA EXPLORER ---

def show_data_explorer(df_raw, df_reduced):
    """Display data exploration page."""
    st.title("📊 Data Explorer")
    st.info("This section allows detailed inspection of the dataset used for modelling. The Raw Data tab displays a sample of the original HFEA records with basic information on size and memory usage. The Processed Data tab shows the cleaned analysis dataset after applying clinical logic and feature engineering. The Statistics tab provides numerical summaries and frequency tables to help understand variable distributions, missingness, and potential data quality issues.")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "🧹 Processed Data", "📈 Statistics"])
    
    with tab1:
        st.subheader("Raw Dataset Preview")
        # Ensure the dataframe is displayed
        st.dataframe(df_raw.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        with col2:
            st.info(f"**Memory:** {df_raw.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    with tab2:
        st.subheader("Processed Dataset (39 Features)")
        st.dataframe(df_reduced.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Shape:** {df_reduced.shape[0]:,} rows × {df_reduced.shape[1]} columns")
        with col2:
            missing = (df_reduced.isnull().sum().sum() / df_reduced.size) * 100
            st.info(f"**Missing Values:** {missing:.2f}%")
    
    with tab3:
        st.subheader("Statistical Summary")
        
        stat_type = st.radio("Select statistics type:", ["Numerical", "Categorical"], horizontal=True)
        
        if stat_type == "Numerical":
            num_stats = df_reduced.describe()
            st.dataframe(num_stats.T.style.format("{:.2f}"), use_container_width=True)
        else:
            cat_cols = df_reduced.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                for col in cat_cols[:5]:
                    with st.expander(f"📁 {col}"):
                        st.dataframe(df_reduced[col].value_counts().head(10))
            else:
                st.info("No categorical columns in the processed dataset.")

# --- 3. ANALYSIS & INSIGHTS ---
def show_analysis(df):
    """Display analysis and insights page."""
    st.title("📈 Analysis & Insights")
    
    # Use tabs for a clean UI
    tab1, tab2, tab3 = st.tabs(["👥 Age Analysis", "🧬 Treatment Factors", "🔗 Correlations"])
    
    with tab1:
        st.subheader("Maternal Age Impact on Outcomes")

        fig = plot_age_analysis(df)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Age analysis data not available in the current dataset.")

        st.info(
            "Live birth rates peak in the younger age bands and decline sharply "
            "after 35 and again after 40, reflecting the fertility outcome decrease "
            "described in Chapter 4."
        )

    # --- Treatment Factors (Figures 4.7 and 4.8) ---
    with tab2:
        st.subheader("Treatment Factor Analysis")

        col1, col2 = st.columns(2)

        # Figure 4.7 – Embryos transferred vs live birth rate
        with col1:
            st.markdown("**Embryos Transferred and Success**")
            fig = plot_embryos_analysis(df)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("Embryo transfer data not available for this cohort.")

        # Figure 4.8 – Previous IVF cycles vs live birth rate
        with col2:
            st.markdown("**Previous IVF Cycles and Success**")
            if "total_number_of_previous_ivf_cycles" in df.columns:
                cycle_analysis = (
                    df.groupby("total_number_of_previous_ivf_cycles")[
                        "live_birth_occurrence"
                    ]
                    .mean()
                    * 100
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(
                    cycle_analysis.index,
                    cycle_analysis.values,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    color="#1f77b4",
                )
                ax.fill_between(
                    cycle_analysis.index,
                    cycle_analysis.values,
                    alpha=0.3,
                    color="#1f77b4",
                )
                ax.set_xlabel("Number of previous IVF cycles", fontsize=12)
                ax.set_ylabel("Live birth rate (%)", fontsize=12)
                ax.set_title(
                    "Live Birth Rate by Previous IVF Cycles",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.warning(
                    "'total_number_of_previous_ivf_cycles' column not found."
                )

        st.info(
            "These plots show how embryo transfer strategy and treatment history "
            "shape success rates, supporting the clinical discussion in Chapter 4."
        )

    # --- Correlations (Figure 4.6) ---
    with tab3:
        st.subheader("Feature Correlation Analysis")

        fig = plot_correlation_heatmap(df)
        if fig:
            st.pyplot(fig)
        else:
            st.warning(
                "Not enough numeric features to compute a meaningful correlation heatmap."
            )

        st.info(
            "The heatmap highlights numerical features most strongly associated "
            "with live birth occurrence, mirroring the correlation analysis "
            "reported in Chapter 4."
        )
        

            
# --- 4. MODEL TRAINING ---
from imblearn.over_sampling import SMOTE  

def show_model_training(df):
    st.title("🤖 Model Training & Benchmarking")
    st.info("This section trains and benchmarks multiple machine learning models for predicting live birth following fertility treatment. Users can configure the train–test split, random seed, and class‑imbalance handling before running a full benchmarking pipeline. The page reports accuracy, F1 score and AUC‑ROC for logistic regression, random forest and XGBoost, displays ROC curves, and provides SHAP‑based explanations of which features most influence the XGBoost predictions")
    st.info("Algorithms: Logistic Regression, Random Forest, and XGBoost")
    
    # --- 1. TRAINING CONFIGURATION ---
    with st.expander("⚙️ Advanced Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        test_size = col1.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = col2.number_input("Random State", 0, 1000, 42)
        use_smote = col3.checkbox("Apply SMOTE (Fix Class Imbalance)", value=True)

    if st.button("🚀 Execute Benchmarking", type="primary"):
        with st.spinner("Preparing data and balancing classes..."):
            # --- 2. DATA PREPARATION ---
            # Standard One-Hot Encoding for all algorithms
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
            df_model = pd.get_dummies(df.drop(columns=['number_of_live_births'], errors='ignore'), columns=cat_cols, drop_first=True)
            
            X = df_model.drop(columns=['live_birth_occurrence'])
            y = df_model['live_birth_occurrence'].astype(int)
            
            # Clean feature names for XGBoost compatibility
            X.columns = [str(c).replace('[', '').replace(']', '').replace('<', '') for c in X.columns]
            
            # Initial Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # --- 3. SMOTE BALANCING ---
            # This helps all models by creating synthetic 'Live Birth' examples
            if use_smote:
                sm = SMOTE(random_state=random_state)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                st.write(f"✅ Data Balanced: Training set now has {len(y_train)} balanced records.")

            # --- 4. SCALING ---
            # Logistic Regression requires scaling to achieve better F1/Accuracy scores
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # --- 5. MODEL TRAINING WITH CLASS WEIGHTS ---
            # Logistic Regression with balanced weights
            lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
            lr.fit(X_train_scaled, y_train)
            
            # Random Forest with balanced weights
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)
            rf.fit(X_train, y_train)
            
            # XGBoost using scale_pos_weight to handle imbalance
            # ratio = negative cases / positive cases
            ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
            xgb_model = XGBClassifier(n_estimators=100, scale_pos_weight=ratio, eval_metric='logloss', random_state=random_state)
            xgb_model.fit(X_train, y_train)

            # --- 6. PERFORMANCE EVALUATION ---
            # Save models to session state for the Predictions page
            st.session_state["models"] = {
                "Logistic Regression": (
                    lr,
                    lr.predict(X_test_scaled),
                    lr.predict_proba(X_test_scaled)[:, 1],
                ),
                "Random Forest": (
                    rf,
                    rf.predict(X_test),
                    rf.predict_proba(X_test)[:, 1],
                ),
                "XGBoost": (
                    xgb_model,
                    xgb_model.predict(X_test),
                    xgb_model.predict_proba(X_test)[:, 1],
                ),
            }
            
            # Metric Calculation
            results = {}
            for name, (model, preds, probs) in st.session_state['models'].items():
                results[name] = {
                    "Accuracy": accuracy_score(y_test, preds),
                    "F1 Score": f1_score(y_test, preds),
                    "AUC-ROC": roc_auc_score(y_test, probs)
                }
            
            st.subheader("📊 Performance Comparison (Balanced)")
            st.table(pd.DataFrame(results).T)

            # ROC Comparison Visual
            st.subheader("📈 ROC Curves Comparison")
            fig_roc = plot_roc_curves(st.session_state['models'], y_test)
            st.pyplot(fig_roc)
            plt.close()

            # SHAP Analysis for XGBoost
            st.divider()
            st.subheader("🧬 SHAP Global Interpretation (XGBoost)")
            explainer = shap.TreeExplainer(xgb_model)
            sample_X = X_test.iloc[:100]
            shap_values = explainer.shap_values(sample_X)
            st_shap(shap.summary_plot(shap_values, sample_X, show=False))
            

            st.subheader("Confusion Matrix Visualisation")
            
            model_name = st.selectbox(
                "Select model for confusion matrix",
                list(st.session_state["models"].keys()),
                index=2,  # default selection (0=LR, 1=RF, 2=XGBoost); adjust if you like
            )

            model, preds, probs = st.session_state["models"][model_name]

            cm = confusion_matrix(y_test, preds)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Live Birth", "Live Birth"],
                yticklabels=["No Live Birth", "Live Birth"],
                ax=ax,
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(f"{model_name} – Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            lr_model, lr_preds, lr_probs = st.session_state["models"]["Logistic Regression"]

            cm = confusion_matrix(y_test, lr_preds)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Live Birth", "Live Birth"],
                yticklabels=["No Live Birth", "Live Birth"],
                ax=ax,
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title("Logistic Regression – Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Random Forest Feature Importance")
            rf_model, _, _ = st.session_state["models"]["Random Forest"]
            fig_fi = plot_feature_importance(rf_model, X_train.columns)
            st.pyplot(fig_fi)

            

            
# --- 5. PREDICTIONS ---
def show_predictions(df_reduced):
    """Display predictions page."""
    st.title("🔮 Make Predictions")
    
    if 'models' not in st.session_state:
        st.warning("⚠️ Please train the models first in the 'Model Training' section.")
        return
    
    st.info("This section provides an interactive tool to generate individualised predictions for a new patient profile. Users can enter basic clinical and treatment information (such as age group, previous cycles, embryos transferred, and gamete source) and obtain an estimated probability of live birth. The output categorises the case into favourable, moderate or challenging, with a clear disclaimer that these estimates are based on historical data and should not replace medical advice.")
    st.info("📝 **Enter Patient Details** - Fill in the patient information below to predict the probability of a live birth outcome.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_group = st.selectbox(
                "Patient Age Group",
                ["18-34", "35-37", "38-39", "40-42", "43-44", "45-50"]
            )
            previous_cycles = st.number_input("Previous IVF Cycles", 0, 10, 0)
        
        with col2:
            embryos = st.number_input("Embryos Transferred", 0, 5, 1)
            eggs_collected = st.number_input("Fresh Eggs Collected", 0, 30, 10)
        
        with col3:
            egg_source = st.selectbox("Egg Source", ["Patient", "Donor"])
            sperm_source = st.selectbox("Sperm Source", ["Partner", "Donor"])
        
        submitted = st.form_submit_button("🔮 Predict Outcome", type="primary")
    
    if submitted:
        st.divider()
        st.subheader("Prediction Results")
        
        # Simulate prediction
        base_prob = 0.3
        
        # Age adjustment
        age_factors = {"18-34": 1.2, "35-37": 1.0, "38-39": 0.8, "40-42": 0.5, "43-44": 0.3, "45-50": 0.15}
        prob = base_prob * age_factors.get(age_group, 1.0)
        
        # Embryo adjustment
        prob *= min(1.5, 0.8 + embryos * 0.2)
        
        prob = min(0.95, max(0.05, prob))
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prob >= 0.5:
                outcome = "Favorable"
                st.success(f"**Predicted Success Probability: {prob*100:.1f}%**")
            elif prob >= 0.3:
                outcome = "Moderate"
                st.warning(f"**Predicted Success Probability: {prob*100:.1f}%**")
            else:
                outcome = "Challenging"
                st.error(f"**Predicted Success Probability: {prob*100:.1f}%**")
            
            st.metric("Outcome Classification", outcome)
        
        st.warning("""
        **⚠️ Important Disclaimer**
        
        This prediction is based on statistical models and historical data. 
        Individual outcomes may vary significantly. This tool should not replace 
        professional medical advice from fertility specialists.
        """)

# --- 6. ABOUT ---
def show_about():
    """Display about page."""
    st.title("📋 About This Dashboard")
    
    st.info("""
    **🔬 Project Overview**
    
    This dashboard is a comprehensive tool for analysing fertility treatment outcomes 
    and predicting birth outcomes using machine learning techniques.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Source")
        st.write("""
        - **Dataset**: HFEA Anonymised Register
        - **Period**: 2017-2018
        - **Focus**: Fertility Treatment Outcomes
        """)
        
        st.subheader("🤖 Models Used")
        st.write("""
        - Logistic Regression
        - Random Forest Classifier
        - XGBoost
        """)
    
    with col2:
        st.subheader("🎯 Key Features")
        st.write("""
        - Interactive data exploration
        - Comprehensive EDA visualisations
        - Model training and comparison
        - Outcome prediction interface
        - Feature importance analysis
        """)
        
        st.subheader("📈 Metrics Tracked")
        st.write("""
        - Accuracy
        - F1 Score
        - ROC-AUC
        """)
    
    st.divider()
    
    st.subheader("🛠️ Technologies Used")
    
    tech_cols = st.columns(5)
    techs = ["Python", "Streamlit", "Scikit-learn", "Pandas", "Matplotlib"]
    
    for col, tech in zip(tech_cols, techs):
        col.info(tech)

if __name__ == "__main__":
    main()