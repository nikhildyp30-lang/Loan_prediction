
# `Project Title: Optimal Loan Amount & Approval Status Prediction`

1. Objective
Build a machine learning system that predicts whether a loan application will be approved/rejected and recommends an optimal loan amount based on applicant’s financial and demographic profile. The goal is to help banks automate initial screening, reduce manual effort, and minimise default risk.

2. Dataset
Source: fin_loan_data.csv – 1,000 loan applications, 12 features.

Targets:

Classification: Loan_Status (Approved / Rejected)

Regression: LoanAmount (optimal amount)

Key features: ApplicantIncome, Credit_History, Dependents, Education, Employment_Type, Property_Area, Loan_Amount_Term, etc.

3. Data Preprocessing & Feature Engineering
Dropped rows with missing LoanAmount.

Imputed missing Employment_Type and Credit_History with mode.

Created new features:

Income_per_Dependent = Income / (Dependents+1)

Monthly_Income = Income / 12

Log_ApplicantIncome = log(1+Income)

Encoded categorical variables (Label Encoding + One‑Hot Encoding) and scaled numeric features.

4. Modelling & Results
Two models were trained (80/20 train/test split):

Model	Accuracy	Precision (Rejected)	Recall (Rejected)	R² (Loan Amount)
Logistic Regression (baseline)	58.0%	26.7%	5.4%	–
Random Forest (baseline)	54.9%	36.2%	23.0%	–
Linear Regression	–	–	–	0.62
Random Forest Regressor	–	–	–	0.71
Best classification model: Random Forest with SMOTE + hyperparameter tuning → accuracy 68.4%, recall for rejections 52%.

Best regression model: Random Forest Regressor – explains 71% of variance in loan amount.

5. Deployment
Built an interactive Streamlit dashboard (app.py) with:

Data overview, statistical summaries, EDA graphs, descriptive/prescriptive insights, and ML predictions with adjustable test size.

Deployed on Streamlit Cloud (https://share.streamlit.io).

Source code hosted on GitHub: nikhilidyp30-lang/Loan_prediction.

All dependencies listed in requirements.txt.

6. Key Business Insights
Credit history is the strongest predictor of approval.

Income drives the loan amount (correlation 0.62).

Graduates and salaried applicants receive higher loans.

Model performance is good for approvals (91% recall) but weaker for rejections – SMOTE improved recall from 5% to 52%.

Recommended deployment: use the Random Forest model with SMOTE, flag borderline cases (confidence 50‑70%) for manual review.
