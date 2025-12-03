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
Please input following details.
""")

# ---------------------------------------------------------
# SECTION 1: Personal Information
# ---------------------------------------------------------

st.header("1️⃣ Personal Information")
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=100, value=30)

    Gender = st.selectbox(
        "Gender",
        cat("Gender")
    )

with col2:
    MaritalStatus = st.selectbox(
        "Marital Status",
        cat("MaritalStatus")
    )

    MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)


# ---------------------------------------------------------
# SECTION 2: Customer Profile
# ---------------------------------------------------------

st.header("2️⃣ Customer Background & Profile")
col3, col4 = st.columns(2)

with col3:
    Occupation = st.selectbox(
        "Occupation",
        cat("Occupation")
    )

    Designation = st.selectbox(
        "Designation",
        cat("Designation")
    )

with col4:
    CityTier = st.selectbox(
        "City Tier",
        sorted(df["CityTier"].unique())
    )

    OwnCar = st.radio("Own Car?", sorted(df["OwnCar"].unique()))
    Passport = st.radio("Passport?", sorted(df["Passport"].unique()))


# ---------------------------------------------------------
# SECTION 3: Travel & Vacation Behavior
# ---------------------------------------------------------

st.header("3️⃣ Travel & Vacation Behavior")
col5, col6 = st.columns(2)

with col5:
    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

with col6:
    NumberOfTrips = st.number_input(
        "Number of Trips per Year",
        min_value=0,
        max_value=20,
        value=1
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star Rating",
        sorted(df["PreferredPropertyStar"].unique())
    )


# ---------------------------------------------------------
# SECTION 4: Sales Interaction Details
# ---------------------------------------------------------

st.header("4️⃣ Sales Interaction Details")
col7, col8 = st.columns(2)

with col7:
    TypeofContact = st.selectbox(
        "Type of Contact",
        cat("TypeofContact")
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        cat("ProductPitched")
    )

with col8:
    DurationOfPitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0.0,
        max_value=60.0,
        value=10.0
    )

    PitchSatisfactionScore = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    NumberOfFollowups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=20,
        value=2
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
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}

st.subheader("📦 Input Data Summary")
st.json(input_data)


# ---------------------------------------------------------
# Prediction Button
# ---------------------------------------------------------

if st.button("Predict"):
    st.success("Prediction logic goes here (connect your model).")

    prediction = model.predict(input_data)[0]
    result = "Customer is likely to Take Product" if prediction == 1 else "Customer will not Take the Product"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")    
