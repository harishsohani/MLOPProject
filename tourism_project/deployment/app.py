import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="harishsohani/MLOP-Project-Tourism", filename="best_tourism_model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism App - Input form")
st.write("""
This application predicts the likelihood of whether a customer would take the product based on following set of parameters.
Please provide the following details.
""")

# ---------------------------------------------------------
# Define Unique Values for each column
# ---------------------------------------------------------

TypeofContact_vals = ['Self Enquiry', 'Company Invited']

Occupation_vals = ['Salaried', 'Free Lancer', 'Small Business', 'Large Business']

Gender_vals = ['Female', 'Male', 'Fe Male']   # You may want to fix "Fe Male"

ProductPitched_vals = ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King']

MaritalStatus_vals = ['Single', 'Divorced', 'Married', 'Unmarried']

Designation_vals = ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP']

CityTier_vals = [1, 2, 3]

PreferredPropertyStar_vals = [3.0, 4.0, 5.0]

NumberOfTrips_vals = [1, 2, 7, 5, 6, 3, 4, 19, 21, 8, 20, 22]

PitchSatisfactionScore_vals = [1, 2, 3, 4, 5]


# ---------------------------------------------------------
# SECTION 1: Personal Information
# ---------------------------------------------------------

st.header("1️⃣ Personal Information")
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=100, value=30)
    Gender = st.selectbox("Gender", Gender_vals)

with col2:
    MaritalStatus = st.selectbox("Marital Status", MaritalStatus_vals)
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)



# ---------------------------------------------------------
# Section 2: Customer Profile
# ---------------------------------------------------------
st.header("2️⃣ Customer Background & Profile")
col3, col4 = st.columns(2)

with col3:
    Occupation = st.selectbox("Occupation", Occupation_vals)
    Designation = st.selectbox("Designation", Designation_vals)

with col4:
    CityTier = st.selectbox("City Tier", sorted(CityTier_vals))
    OwnCar_display = st.radio("Own Car?", ["Yes", "No"])
    Passport_display = st.radio("Passport?", ["Yes", "No"])

# Convert Yes/No → 1/0  
OwnCar = 1 if OwnCar_display == "Yes" else 0
Passport = 1 if Passport_display == "Yes" else 0

# ---------------------------------------------------------
# SECTION 3: Travel & Vacation Behavior
# ---------------------------------------------------------

st.header("3️⃣ Travel & Vacation Behavior")
col5, col6 = st.columns(2)

with col5:
    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=2
    )
    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting", min_value=0, max_value=10, value=0
    )

with col6:
    NumberOfTrips = st.number_input(
      "Number of Trips per Year",
      min_value=0,
      max_value=50,
      value=1
    )
    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star", PreferredPropertyStar_vals
    )


# ---------------------------------------------------------
# SECTION 4: Sales Interaction Details
# ---------------------------------------------------------

st.header("4️⃣ Sales Interaction Details")
col7, col8 = st.columns(2)

with col7:
    TypeofContact = st.selectbox("Type of Contact", TypeofContact_vals)
    ProductPitched = st.selectbox("Product Pitched", ProductPitched_vals)

with col8:
    DurationOfPitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=10.0
    )
    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score", sorted(PitchSatisfactionScore_vals)
    )
    NumberOfFollowups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=20, value=2
    )


# ---------------------------------------------------------
# Prepare Input for Model
# ---------------------------------------------------------

input_data = {
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,        # now 0/1
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,            # now 0/1
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}

import_data_df = pd.DataFrame([input_data])

st.subheader("📦 Input Data Summary")
st.json(input_data)


# ---------------------------------------------------------
# Prediction Button
# ---------------------------------------------------------

if st.button("Predict"):
    st.success("Prediction logic goes here (connect your model).")

    prediction = model.predict(import_data_df)[0]
    result = "Customer is likely to Take Product" if prediction == 1 else "Customer will not Take the Product"
    st.subheader("Prediction Result:")
    st.success(f"Prediction as per Model: **{result}**")    
