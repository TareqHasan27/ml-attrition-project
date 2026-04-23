import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()


st.title("👔 Employee Attrition Predictor")
st.markdown("""
This app predicts whether an employee is likely to **leave or stay** 
based on their profile. Built using **SVM** — the best performing model 
among Decision Tree, Random Forest, SVM, and Naive Bayes.
""")
st.divider()





st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
distance_from_home = st.sidebar.slider("Distance From Home (km)", 1, 29, 10)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
years_with_curr_manager = st.sidebar.slider("Years With Current Manager", 0, 17, 3)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
job_satisfaction = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4], 
                                         format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x])
work_life_balance = st.sidebar.selectbox("Work Life Balance", [1, 2, 3, 4],
                                          format_func=lambda x: {1:"Bad", 2:"Good", 3:"Better", 4:"Best"}[x])
overtime = st.sidebar.selectbox("OverTime", [0, 1], 
                                 format_func=lambda x: "Yes" if x == 1 else "No")
gender = st.sidebar.selectbox("Gender", [0, 1],
                               format_func=lambda x: "Male" if x == 1 else "Female")
marital_status = st.sidebar.selectbox("Marital Status", [0, 1, 2],
                                       format_func=lambda x: {0:"Divorced", 1:"Married", 2:"Single"}[x])




def build_input():
    input_data = {
        'Age': age,
        'BusinessTravel': 1,
        'DailyRate': 800,
        'Department': 1,
        'DistanceFromHome': distance_from_home,
        'Education': 3,
        'EducationField': 1,
        'EnvironmentSatisfaction': 3,
        'Gender': gender,
        'HourlyRate': 66,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'JobRole': 1,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': 14000,
        'NumCompaniesWorked': 2,
        'OverTime': overtime,
        'PercentSalaryHike': 14,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': 3,
        'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': years_with_curr_manager
    }
    return pd.DataFrame([input_data])




col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Profile Summary")
    st.write(f"**Age:** {age}")
    st.write(f"**Monthly Income:** ${monthly_income:,}")
    st.write(f"**Years at Company:** {years_at_company}")
    st.write(f"**OverTime:** {'Yes' if overtime == 1 else 'No'}")
    st.write(f"**Job Satisfaction:** {job_satisfaction}/4")
    st.write(f"**Work Life Balance:** {work_life_balance}/4")

with col2:
    st.subheader("Prediction Result")
    
    if st.button("🔍 Predict Attrition", use_container_width=True):
        input_df = build_input()
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.decision_function(input_scaled)[0]
        
        if prediction == 1:
            st.error("⚠️ HIGH RISK — This employee is likely to LEAVE")
            st.markdown("**Suggested Actions:**")
            st.markdown("- Consider salary revision")
            st.markdown("- Review overtime workload")
            st.markdown("- Schedule career growth discussion")
        else:
            st.success("✅ LOW RISK — This employee is likely to STAY")
            st.markdown("**Keep doing:**")
            st.markdown("- Maintain current work environment")
            st.markdown("- Continue growth opportunities")
            
            


st.divider()
st.subheader("📊 Model Performance")

col3, col4, col5, col6 = st.columns(4)
col3.metric("Decision Tree", "79.93%")
col4.metric("Naive Bayes", "84.35%")
col5.metric("Random Forest", "88.10%")
col6.metric("SVM (Best)", "88.78%")

st.info("ℹ️ SVM was selected as the best model based on highest test accuracy.")




st.divider()
st.subheader("🔢 Confusion Matrix")
st.image('confusion_matrix.png', 
         caption='Confusion Matrix of Best Model (SVM)',
         width=500)

st.markdown("""
**How to read this:**
- **Top-left (255):** Correctly predicted employees who stayed ✅
- **Bottom-right:** Correctly predicted employees who left ✅  
- **Other cells:** Wrong predictions ❌
""")

