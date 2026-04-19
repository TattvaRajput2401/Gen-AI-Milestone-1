import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Gurgaon Real Estate Analytics", page_icon="🏡", layout="wide")
st.title("🏡 Intelligent Property Price Predictor (Gurgaon)")
st.markdown("Enter the property specifications below to generate an AI-driven valuation.")

# 2. Load the Pipeline
@st.cache_resource
def load_model():
    return joblib.load('rf_gurgaon_pipeline_final.pkl')

pipeline = load_model()

# 3. User Interface Layout
st.header("Property Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    property_type = st.selectbox("Property Type", ['flat', 'house'])
    sector = st.text_input("Sector (e.g., 'sector 36', 'sector 89')", value='sector 36').lower()
    built_up_area = st.number_input("Built-up Area (Sq. Ft.)", min_value=100.0, value=1200.0, step=50.0)
    agePossession = st.selectbox("Age of Property", ['New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction'])

with col2:
    bedRoom = st.number_input("Bedrooms", min_value=1.0, value=3.0, step=1.0)
    bathroom = st.number_input("Bathrooms", min_value=1.0, value=2.0, step=1.0)
    balcony = st.selectbox("Balconies", ['0', '1', '2', '3', '3+'])
    floor_category = st.selectbox("Floor Category", ['Low Floor', 'Mid Floor', 'High Floor'])

with col3:
    luxury_category = st.selectbox("Luxury Category", ['Low', 'Medium', 'High'])
    furnishing_type = st.selectbox("Furnishing (0=Unfurnished, 1=Semi, 2=Fully)", [0.0, 1.0, 2.0])
    servant_room = st.selectbox("Servant Room", [0.0, 1.0])
    store_room = st.selectbox("Store Room", [0.0, 1.0])

# 4. Prediction Engine
if st.button("Generate Valuation", type="primary"):
    # Map inputs exactly to the training dataframe structure
    input_data = pd.DataFrame([[
        property_type, sector, bedRoom, bathroom, balcony, 
        agePossession, built_up_area, servant_room, store_room, 
        furnishing_type, luxury_category, floor_category
    ]], columns=[
        'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 
        'agePossession', 'built_up_area', 'servant room', 'store room', 
        'furnishing_type', 'luxury_category', 'floor_category'
    ])
    
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success("Analytics Engine execution complete.")
        st.metric(label="Estimated Market Value", value=f"₹ {prediction:.2f} Crores")
    except Exception as e:
        st.error(f"Error processing prediction: {e}")

if st.button("Generate Report", type="secondary"):
    # Map inputs exactly to the training dataframe structure
    input_data = pd.DataFrame([[
        property_type, sector, bedRoom, bathroom, balcony, 
        agePossession, built_up_area, servant_room, store_room, 
        furnishing_type, luxury_category, floor_category
    ]], columns=[
        'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 
        'agePossession', 'built_up_area', 'servant room', 'store room', 
        'furnishing_type', 'luxury_category', 'floor_category'
    ])
    
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success("Analytics Engine execution complete.")
        st.metric(label="Estimated Market Value", value=f"₹ {prediction:.2f} Crores")
    except Exception as e:
        st.error(f"Error processing prediction: {e}")
