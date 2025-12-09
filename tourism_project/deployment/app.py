import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="samdurai102024/Tourism-Package-Prediction", filename="best_tourism_purchase_prediction_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Tourism Package Purchase Prediction App submitted by Sathiamurthy Samidurai PGAIML student")
st.write("""
This application predicts the likelihood of a Tourism Package Purchase based on its operational parameters.
Please enter the configuration data below to get a prediction.
""")

# User input
st.sidebar.header("User Input Features")
Age = st.number_input("Age", min_value = 18, max_value = 100, value = 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.number_input("City Tier", min_value = 1, max_value = 5, value = 1)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value = 1, max_value = 180)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=100, value = 2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=100, value = 5)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value = 1, max_value = 5, value = 3)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=100, value = 5)
Passport = st.selectbox("Passport", ["1", "0"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value = 1, max_value = 5, value = 2)
OwnCar = st.selectbox("Own Car", ["1", "0"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value = 2)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=200000, value = 80000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
}])


if st.button("Predict Tourism Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction == 1 else "No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts that user will purchase tourism package: **{result}**")
