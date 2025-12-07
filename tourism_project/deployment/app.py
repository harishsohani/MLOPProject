
# import streamlit library for IO
import streamlit as st

# import pandas
import pandas as pd

# library to download fine from Hugging Face
from huggingface_hub import hf_hub_download

# library to load model
import joblib




# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tourism Prediction App",
    layout="wide"
)
# Streamlit UI for Machine Failure Prediction
#st.title("Tourism App - Input form for Predection")
st.write("""
This application predicts the likelihood of whether a customer would take the product based on following set of parameters.
Please provide the following details.
""")



# ---------------------------------------------------------
# LIGHT CSS OPTIMIZATION
# ---------------------------------------------------------
st.markdown("""
<style>
/* Reduce page padding */
.block-container {
    padding-top: 0.25rem; /* smaller padding on top */
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Reduce vertical gaps between widgets */
div[data-testid="stVerticalBlock"] {
    row-gap: 0.5rem;
}

/* Tighter expander headers */
.streamlit-expanderHeader {
    font-size: 1rem;
    padding: 0.4rem 0.5rem;
}

/* section header */
.section-header {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #333333 !important;
    margin-top: 20px !important;
}
</style>
""", unsafe_allow_html=True)


# Download and load the model
model_path = hf_hub_download(
    repo_id="harishsohani/MLOP-Project-Tourism", 
    filename="best_tourism_model.joblib"
    )
model = joblib.load(model_path)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🏖️ Tourism Purchase Prediction App")
st.write("Fill in the details below and click **Predict** to see if the customer is likely to purchase the product.")



# ---------------------------------------------------------
# DROPDOWN VALUES
#
# Define predefines set values for each input applicable
# These are used to show pick list
# ---------------------------------------------------------
TypeofContact_vals = ['Self Enquiry', 'Company Invited']

Occupation_vals = ['Salaried', 'Free Lancer', 'Small Business', 'Large Business']

Gender_vals = ['Female', 'Male']

ProductPitched_vals = ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King']

MaritalStatus_vals = ['Single', 'Divorced', 'Married', 'Unmarried']

Designation_vals = ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP']

CityType = [ "Tier 1", "Tier 2", "Tier3"]

CityTier_vals = [1, 2, 3]

PreferredPropertyStar_vals = [3.0, 4.0, 5.0]

NumberOfTrips_vals = [1, 2, 7, 5, 6, 3, 4, 19, 21, 8, 20, 22]

PitchSatisfactionScore_vals = [1, 2, 3, 4, 5]



# ---------------------------------------------------------
# PERSONAL INFORMATION
# ---------------------------------------------------------
with st.expander("👤 1. Personal and Professional Information", expanded=True):

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Age = st.number_input("Age", 18, 120, 30)
        Gender = st.selectbox("Gender", Gender_vals)

    with col2:
        MaritalStatus = st.selectbox("Marital Status", MaritalStatus_vals)
        CityTier_label = st.selectbox("City Tier", CityType)

    with col3:
        OwnCar_display = st.radio("Own Car?", ["Yes", "No"])
        Passport_display = st.radio("Has Passport?", ["Yes", "No"])

    with col4:
        Occupation = st.selectbox("Occupation", Occupation_vals)
        Designation = st.selectbox("Designation", Designation_vals)

    with col5:
        MonthlyIncome = st.number_input("Monthly Income (₹)", 0, 1000000, 100000)

CityTier = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}[CityTier_label]
OwnCar = 1 if OwnCar_display == "Yes" else 0
Passport = 1 if Passport_display == "Yes" else 0


# ---------------------------------------------------------
# TRAVEL INFORMATION
# ---------------------------------------------------------
with st.expander("✈️ 2. Travel Information", expanded=False):

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        NumberOfTrips = st.number_input("Average Trips per Year", 0, 50, 2)

    with col2:
        NumberOfPersonVisiting = st.number_input("Total Persons Visiting", 1, 10, 2)

    with col3:
        NumberOfChildrenVisiting = st.number_input("Children (Below 5 yrs)", 0, 10, 0)

    with col4:
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])


# ---------------------------------------------------------
# INTERACTION INFORMATION
# ---------------------------------------------------------
with st.expander("🗣️ 3. Interaction Details", expanded=False):

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        TypeofContact = st.selectbox("Type of Contact", TypeofContact_vals)

    with col2:
        ProductPitched = st.selectbox("Product Pitched", ProductPitched_vals)

    with col3:
        DurationOfPitch = st.number_input("Pitch Duration (minutes)", 0, 200, 10)

    with col4:
        NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 50, 1)

    with col5:
        PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [5, 4, 3, 2, 1])



# --------------------------
# Prepare input data frame
# ------------------------
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

input_df = pd.DataFrame([input_data])


# ---------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------
st.markdown("---")
if st.button("🔍 Predict", use_container_width=True):

    prediction = model.predict(input_df)[0]
    result = "Customer is **likely** to purchase the product." if prediction == 1 \
             else "Customer is **unlikely** to purchase the product."

    st.success(result)
