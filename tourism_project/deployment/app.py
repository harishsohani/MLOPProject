
# import streamlit library for IO
import streamlit as st

# import pandas
import pandas as pd

# library to download fine from Hugging Face
from huggingface_hub import hf_hub_download

# library to load model
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="harishsohani/MLOP-Project-Tourism", filename="best_tourism_model.joblib")
model = joblib.load(model_path)


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



st.set_page_config(page_title="Tourism App – Input Form", layout="wide")

# ----- COMPACT CSS -----
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #1a73e8;
}

h1 {
    font-size: 26px !important;
    margin-bottom: 8px;
}

label, .stSelectbox label, .stNumberInput label {
    font-size: 14px !important;
    font-weight: 600 !important;
}

.stButton > button {
    background-color: #1a73e8;
    color: white;
    padding: 10px 20px;
    font-size: 16px;
    width: 40%;
    border-radius: 8px;
}

.result-box {
    padding: 10px;
    background-color: #e8f4ff;
    border-left: 4px solid #1a73e8;
    font-size: 16px;
    border-radius: 6px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.markdown("<h1 style='text-align:center;'>🏖️ Tourism App – Customer Input</h1>", unsafe_allow_html=True)


# -----------------------
# LAYOUT (2-COLUMN PAGE)
# -----------------------
left, right = st.columns(2)


# ===== LEFT SIDE =====
with left:

    # ---- Personal Information ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>1️⃣ Personal Information</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 0, 120, 30)
        MonthlyIncome = st.number_input("Monthly Income", 0, 500000, 50000)
    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Customer Profile ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>2️⃣ Customer Profile</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        Occupation = st.selectbox("Occupation", ["Salaried", "Business", "Retired"])
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager"])
    with col2:
        CityTier_display = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        Passport_display = st.selectbox("Passport", ["Yes", "No"])

    st.markdown("</div>", unsafe_allow_html=True)


# ===== RIGHT SIDE =====
with right:

    # ---- Travel Behavior ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>3️⃣ Travel Behavior</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        NumberOfTrips = st.number_input("Travel Persons", 1, 10, 2)
        NumberOfChildrenVisiting = st.number_input("Children", 0, 10, 0)
    with col2:
        duration = st.number_input("Trip Days", 1, 60, 5)
        property_type = st.selectbox("Preferred Stay", ["Hotel", "Resort", "Homestay"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Sales Details ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>4️⃣ Sales Interaction</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Premium"])
    with col2:
        DurationOfPitch = st.number_input("Pitch Duration (min)", 0, 120, 30)

    st.markdown("</div>", unsafe_allow_html=True)


# ===== PREDICT BUTTON CENTERED =====
st.write("")
btn_col = st.columns([3, 2, 3])  # center alignment

with btn_col[1]:
    predict = st.button("🔮 Predict", use_container_width=True)

#if predict:
#    st.markdown("<div class='result-box'>Prediction output goes here.</div>", unsafe_allow_html=True)
