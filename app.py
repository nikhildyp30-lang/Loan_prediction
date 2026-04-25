import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import joblib
import io
import base64
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Loan Prediction Dashboard", layout="wide")
st.title("🏦 Loan Approval & Optimal Amount Prediction")
st.markdown("---")

# ------------------- SIDEBAR CONFIGURATION -------------------
st.sidebar.header("⚙️ Configuration")
layout_option = st.sidebar.radio("Select Layout", ["Horizontal (Tabs)", "Vertical (Sidebar Menus)"])

st.sidebar.subheader("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    # Sample data (first few rows of the original dataset)
    sample_data = """Loan_ID,Gender,Married,Dependents,Education,Employment_Type,ApplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status
L1000,Female,Yes,2,Not Graduate,Self-employed,103086,262678.0,360,0.0,Semiurban,Rejected
L1001,Female,No,3,Graduate,Self-employed,56343,424635.0,360,1.0,Urban,Approved
L1002,Female,Yes,0,Not Graduate,Self-employed,38254,384767.0,180,1.0,Rural,Rejected
L1003,Male,No,2,Not Graduate,Salaried,81273,262243.0,180,1.0,Semiurban,Rejected
L1004,Female,Yes,0,Not Graduate,Salaried,79611,409553.0,240,1.0,Urban,Approved"""
    from io import StringIO

    df = pd.read_csv(StringIO(sample_data))
    st.sidebar.info("Using sample data. Upload your own CSV for full analysis.")

# ------------------- LAYOUT HANDLING -------------------
if layout_option == "Horizontal (Tabs)":
    tabs = st.tabs(["📊 Data Overview", "📈 Statistical Analysis", "🔍 Descriptive & Prescriptive", "📉 EDA Graphs",
                    "🤖 ML Predictions"])
    data_tab, stats_tab, desc_tab, eda_tab, ml_tab = tabs
else:
    analysis_choice = st.sidebar.radio("Select Analysis",
                                       ["Data Overview", "Statistical Analysis", "Descriptive & Prescriptive",
                                        "EDA Graphs", "ML Predictions"])
    data_tab = st.container()
    stats_tab = st.container()
    desc_tab = st.container()
    eda_tab = st.container()
    ml_tab = st.container()

# ------------------- DATA OVERVIEW -------------------
with data_tab if layout_option == "Horizontal (Tabs)" else st.container():
    if layout_option == "Vertical (Sidebar Menus)" and analysis_choice != "Data Overview":
        st.empty()
    else:
        st.header("📋 Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 5 rows")
            st.dataframe(df.head())
        with col2:
            st.subheader("Last 5 rows")
            st.dataframe(df.tail())

        st.subheader("Dataset Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.DataFrame({"No missing values": [0]}, index=[0]))

        st.subheader("Data Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Dtype"}))

# ------------------- STATISTICAL ANALYSIS -------------------
with stats_tab if layout_option == "Horizontal (Tabs)" else st.container():
    if layout_option == "Vertical (Sidebar Menus)" and analysis_choice != "Statistical Analysis":
        st.empty()
    else:
        st.header("📈 Statistical Analysis")
        st.subheader("Descriptive Statistics (Numeric Features)")
        st.dataframe(df.describe())

        st.subheader("Categorical Feature Counts")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != 'Loan_ID':
                st.write(f"**{col}**")
                st.dataframe(df[col].value_counts())

# ------------------- DESCRIPTIVE & PRESCRIPTIVE -------------------
with desc_tab if layout_option == "Horizontal (Tabs)" else st.container():
    if layout_option == "Vertical (Sidebar Menus)" and analysis_choice != "Descriptive & Prescriptive":
        st.empty()
    else:
        st.header("🔍 Descriptive & Prescriptive Analysis")
        st.subheader("Key Insights from Data")
        st.markdown("""
        - **Target Variable:** `Loan_Status` (Approved/Rejected). The dataset shows a slight imbalance (more approved).
        - **Income vs Loan Amount:** Positive correlation; higher income tends to get higher loan amounts.
        - **Credit History:** Critical factor – applicants with good credit history are more likely to be approved.
        - **Education:** Graduates tend to receive higher loan amounts.
        - **Property Area:** Urban applicants get higher loans on average.
        """)
        st.subheader("📌 Prescriptive Recommendations")
        st.markdown("""
        **For Banks / Financial Institutions:**
        1. **Prioritize Credit History** – use it as a primary decision factor.
        2. **Consider Income per Dependent** – a better measure of repayment capacity.
        3. **Use Machine Learning** – automate initial screening to reduce manual effort.
        4. **Balance Rejected Cases** – apply SMOTE to improve recall for rejections.
        5. **Implement a Confidence Threshold** – flag borderline cases (confidence 50-70%) for manual review.
        """)

# ------------------- EDA GRAPHS -------------------
with eda_tab if layout_option == "Horizontal (Tabs)" else st.container():
    if layout_option == "Vertical (Sidebar Menus)" and analysis_choice != "EDA Graphs":
        st.empty()
    else:
        st.header("📉 Exploratory Data Analysis")

        # Loan Amount Distribution
        if 'LoanAmount' in df.columns:
            st.subheader("Distribution of Loan Amount")
            fig, ax = plt.subplots()
            df['LoanAmount'].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title("Loan Amount Distribution")
            st.pyplot(fig)

        # Income vs Loan Amount
        if 'ApplicantIncome' in df.columns and 'LoanAmount' in df.columns:
            st.subheader("Applicant Income vs Loan Amount")
            fig, ax = plt.subplots()
            ax.scatter(df['ApplicantIncome'], df['LoanAmount'], alpha=0.5)
            ax.set_xlabel("Applicant Income")
            ax.set_ylabel("Loan Amount")
            st.pyplot(fig)

        # Education vs Loan Amount
        if 'Education' in df.columns and 'LoanAmount' in df.columns:
            st.subheader("Loan Amount by Education Level")
            fig, ax = plt.subplots()
            sns.boxplot(x='Education', y='LoanAmount', data=df, ax=ax)
            st.pyplot(fig)

        # Credit History vs Loan Amount
        if 'Credit_History' in df.columns and 'LoanAmount' in df.columns:
            st.subheader("Loan Amount by Credit History")
            fig, ax = plt.subplots()
            sns.violinplot(x='Credit_History', y='LoanAmount', data=df, ax=ax)
            st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns for correlation.")

# ------------------- MACHINE LEARNING PREDICTIONS -------------------
with ml_tab if layout_option == "Horizontal (Tabs)" else st.container():
    if layout_option == "Vertical (Sidebar Menus)" and analysis_choice != "ML Predictions":
        st.empty()
    else:
        st.header("🤖 Machine Learning Predictions")
        ml_df = df.copy()

        if 'Loan_Status' not in ml_df.columns:
            st.error("Loan_Status column not found. Cannot proceed.")
        else:
            # ---- DATA CLEANING ----
            # Drop rows where target is missing (LoanAmount not needed for classification)
            ml_df = ml_df.dropna(subset=['Loan_Status'])

            # Fill missing values in Employment_Type with mode
            if 'Employment_Type' in ml_df.columns:
                mode_emp = ml_df['Employment_Type'].mode()
                if not mode_emp.empty:
                    ml_df['Employment_Type'] = ml_df['Employment_Type'].fillna(mode_emp[0])

            # Fill missing values in Credit_History with mode
            if 'Credit_History' in ml_df.columns:
                # First ensure numeric, coerce errors to NaN
                ml_df['Credit_History'] = pd.to_numeric(ml_df['Credit_History'], errors='coerce')
                mode_ch = ml_df['Credit_History'].mode()
                if not mode_ch.empty:
                    ml_df['Credit_History'] = ml_df['Credit_History'].fillna(mode_ch[0])
                # Now convert to int (no NaNs left)
                ml_df['Credit_History'] = ml_df['Credit_History'].astype(int)

            # Fill missing Loan_Amount_Term (if any) with mode
            if 'Loan_Amount_Term' in ml_df.columns:
                ml_df['Loan_Amount_Term'] = pd.to_numeric(ml_df['Loan_Amount_Term'], errors='coerce')
                mode_term = ml_df['Loan_Amount_Term'].mode()
                if not mode_term.empty:
                    ml_df['Loan_Amount_Term'] = ml_df['Loan_Amount_Term'].fillna(mode_term[0])
                ml_df['Loan_Amount_Term'] = ml_df['Loan_Amount_Term'].astype(int)

            # Drop remaining rows with any NaN (safety)
            ml_df = ml_df.dropna()

            # ---- FEATURE ENGINEERING ----
            ml_df['Income_per_Dependent'] = ml_df['ApplicantIncome'] / (ml_df['Dependents'] + 1)
            ml_df['Monthly_Income'] = ml_df['ApplicantIncome'] / 12
            ml_df['Log_ApplicantIncome'] = np.log1p(ml_df['ApplicantIncome'])

            # ---- ENCODING CATEGORICALS ----
            le = LabelEncoder()
            cat_cols = ['Gender', 'Married', 'Education', 'Employment_Type', 'Property_Area']
            for col in cat_cols:
                if col in ml_df.columns:
                    ml_df[col] = le.fit_transform(ml_df[col].astype(str))

            # Encode target
            ml_df['Loan_Status'] = le.fit_transform(ml_df['Loan_Status'])  # 0=Approved, 1=Rejected

            # ---- FEATURES & TARGET ----
            features = [c for c in ml_df.columns if c not in ['Loan_ID', 'Loan_Status', 'LoanAmount']]
            X = ml_df[features]
            y = ml_df['Loan_Status']

            # Ensure no NaN in features
            X = X.fillna(0)

            st.subheader("Model Training")
            test_size = st.slider("Test Set Percentage", 10, 40, 20) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model choice
            model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Evaluation
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("Precision", f"{prec:.2%}")
            col3.metric("Recall", f"{rec:.2%}")
            col4.metric("F1-Score", f"{f1:.2%}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())

            # Download model
            if st.button("💾 Download Trained Model (joblib)"):
                model_bytes = io.BytesIO()
                joblib.dump(model, model_bytes)
                model_bytes.seek(0)
                b64 = base64.b64encode(model_bytes.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="loan_model.pkl">Click here to download</a>'
                st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit | Loan Prediction Dashboard")