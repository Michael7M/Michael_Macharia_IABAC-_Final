import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_path = r"C:\Users\Michael\Notebooks\IABAC\Michael_ Macharia_IABAC_Project INX Future Inc_performance_rating_prediction_model_Final Project\performance_rating_prediction_model.pkl"
performance_rating_predictor = pickle.load(open(model_path, 'rb'))

# Load the model using the correct file path


# Your existing Streamlit app code continues here...


# Input fields for the features

# Numerical inputs
age = st.number_input("Enter Age:")
distancefromhome = st.number_input("Enter Distance From Home:")
empeducationlevel = st.number_input("Enter Emp Education Level:")
empenvironmentsatisfaction = st.number_input("Enter Emp Environment Satisfaction:")
emphourlyrate = st.number_input("Enter Emp Hourly Rate:")
empjobinvolvement = st.number_input("Enter Emp Job Involvement:")
empjoblevel = st.number_input("Enter Emp Job Level:")
empjobsatisfaction = st.number_input("Enter Emp Job Satisfaction:")
numcompaniesworked = st.number_input("Enter Number of Companies Worked:")
emplastsalaryhikepercent = st.number_input("Enter Emp Last Salary Hike Percent:")
emprelationshipsatisfaction = st.number_input("Enter Emp Relationship Satisfaction:")
totalworkexperienceinyears = st.number_input("Enter Total Work Experience In Years:")
trainingtimeslastyear = st.number_input("Enter Training Times Last Year:")
empworklifebalance = st.number_input("Enter Emp Work Life Balance:")
experienceyearsatthiscompany = st.number_input("Enter Experience Years At This Company:")
experienceyearsincurrentrole = st.number_input("Enter Experience Years In Current Role:")
yearssincelastpromotion = st.number_input("Enter Years Since Last Promotion:")
yearswithcurrmanager = st.number_input("Enter Years With Current Manager:")

# Categorical inputs using dropdown menus
gender = st.selectbox("Enter Gender:", ['Male', 'Female', 'Other'])
educationbackground = st.selectbox("Enter Education Background:", ['Bachelors', 'Masters', 'PhD', 'Diploma', 'Other'])
maritalstatus = st.selectbox("Enter Marital Status:", ['Single', 'Married', 'Divorced'])
empdepartment = st.selectbox("Enter Emp Department:", ['Sales', 'HR', 'R&D', 'IT', 'Finance', 'Admin'])
empjobrole = st.selectbox("Enter Emp Job Role:", ['Manager', 'Analyst', 'Sales Executive', 'Research Scientist', 'Developer'])
businesstravelfrequency = st.selectbox("Enter Business Travel Frequency:", ['Rarely', 'Frequently', 'Non-Travel'])
overtime = st.selectbox("Enter Overtime:", ['Yes', 'No'])
attrition = st.selectbox("Enter Attrition:", ['Yes', 'No'])

# Collect input in a dictionary
manual_test_input = {
    'age': [age],
    'gender': [gender],
    'educationbackground': [educationbackground],
    'maritalstatus': [maritalstatus],
    'empdepartment': [empdepartment],
    'empjobrole': [empjobrole],
    'businesstravelfrequency': [businesstravelfrequency],
    'distancefromhome': [distancefromhome],
    'empeducationlevel': [empeducationlevel],
    'empenvironmentsatisfaction': [empenvironmentsatisfaction],
    'emphourlyrate': [emphourlyrate],
    'empjobinvolvement': [empjobinvolvement],
    'empjoblevel': [empjoblevel],
    'empjobsatisfaction': [empjobsatisfaction],
    'numcompaniesworked': [numcompaniesworked],
    'overtime': [overtime],
    'emplastsalaryhikepercent': [emplastsalaryhikepercent],
    'emprelationshipsatisfaction': [emprelationshipsatisfaction],
    'totalworkexperienceinyears': [totalworkexperienceinyears],
    'trainingtimeslastyear': [trainingtimeslastyear],
    'empworklifebalance': [empworklifebalance],
    'experienceyearsatthiscompany': [experienceyearsatthiscompany],
    'experienceyearsincurrentrole': [experienceyearsincurrentrole],
    'yearssincelastpromotion': [yearssincelastpromotion],
    'yearswithcurrmanager': [yearswithcurrmanager],
    'attrition': [attrition]
}

# Convert to DataFrame
df_manual_test = pd.DataFrame(manual_test_input)

# Encode categorical variables
numerical_cols = df_manual_test._get_numeric_data().columns
cols = df_manual_test.columns
categorical_cols = list(set(cols) - set(numerical_cols))

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical values
for col in categorical_cols:
    df_manual_test[col] = label_encoder.fit_transform(df_manual_test[col])

# Make a prediction using the model
if st.button("Predict Performance Rating"):
    predicted_performance_rating = performance_rating_predictor.predict(df_manual_test)
    st.write(f"The predicted performance rating is: {predicted_performance_rating[0]}")
