
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

# Define predefines set values for each input applicable
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
# UI OPTIMIZATION (CSS + Layout Tweaks)
# ---------------------------------------------------------
st.markdown("""
    <style>
        /* Reduce padding at top/bottom */
        .main {
            padding-top: 1rem;
        }

        /* Card-style containers */
        .card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }

        /* Smaller headers */
        h1 { font-size: 32px !important; }
        h2 { font-size: 26px !important; }
        h3 { font-size: 20px !important; }

        /* Input element spacing */
        .stSelectbox, .stNumberInput, .stTextInput {
            margin-bottom: -10px;
        }

        /* Prediction box sticky to top-right */
        .sticky {
            position: fixed;
            top: 80px;
            right: 20px;
            width: 300px;
            z-index: 999;
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PERSONAL INFORMATION
# ---------------------------------------------------------
with st.expander("👤 1. Personal and Professional Information", expanded=True):
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age", 18, 90, 30)
        Gender = st.selectbox("Gender", Gender_vals)
        MaritalStatus = st.selectbox("Marital Status", MaritalStatus_vals)

    with col2:
        CityTier_label = st.selectbox("City Tier", CityType)
        #OwnCar = st.selectbox("Owns a Car?", [0, 1])
        #Passport = st.selectbox("Has Passport?", [0, 1])
        OwnCar_display = st.radio("Own Car?", ["Yes", "No"])
        Passport_display = st.radio("Has Passport?", ["Yes", "No"])

    with col3:
        Occupation = st.selectbox("Occupation", Occupation_vals)
        Designation = st.selectbox("Designation", Designation_vals)        
        MonthlyIncome = st.number_input("Monthly Income (₹)", 0, 500000, 50000)

    st.markdown('</div>', unsafe_allow_html=True)

#convert City Tier to numeric
CityTier = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}[CityTier_label]

# Convert Yes/No → 1/0
OwnCar = 1 if OwnCar_display == "Yes" else 0
Passport = 1 if Passport_display == "Yes" else 0


# ---------------------------------------------------------
# TRAVEL INFORMATION
# ---------------------------------------------------------
with st.expander("✈️ 2. Travel Information"):
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        NumberOfTrips = st.number_input("Average Trips per Year", 0, 100, 2)
        NumberOfChildrenVisiting = st.number_input("Children (Below 5 years)", 0, 10, 0)

    with col2:
        NumberOfPersonVisiting = st.number_input("Total Persons Visiting", 1, 10, 2)
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# INTERACTION INFORMATION
# ---------------------------------------------------------
with st.expander("🗣️ 3. Interaction Details"):
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        ProductPitched = st.selectbox("Product Pitched", ProductPitched_vals)


    with col2:
        DurationOfPitch = st.number_input("Pitch Duration (minutes)", 0, 200, 10)
        NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 50, 1)

    with col3:
        PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    
    st.markdown('</div>', unsafe_allow_html=True)

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
# Predict Button
# ---------------------------------------------------------
if st.button("🔮 Predict"):
    # ---------------------------------------------------------
    # Prediction Button
    # ---------------------------------------------------------
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        result = (
            "Customer is likely to purchase the product"
            if prediction == 1 else
            "Customer is unlikely to purchase the product"
        )
        
        st.subheader("Prediction Result")
        st.success(f"**{result}**")
        
            
    '''pred = model.predict(df_input)[0]
    st.markdown(f"""
        <div class="sticky">
            <h2>📈 Prediction: {pred}</h2>
        </div>
    """, unsafe_allow_html=True)'''
