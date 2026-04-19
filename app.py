import streamlit as st
import pandas as pd
import joblib
import os

from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
from pydantic import BaseModel

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Gurgaon Real Estate Analytics", page_icon="🏡", layout="wide")
st.title("🏡 Intelligent Property Price Predictor (Gurgaon)")
st.markdown("Enter the property specifications below to generate an AI-driven valuation.")

# -------------------------------
# 2. LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load('rf_gurgaon_pipeline_final.pkl')

pipeline = load_model()

# -------------------------------
# 3. DEFINE TOOL FUNCTION
# -------------------------------
def predict_property_price(
    property_type: str,
    sector: str,
    bedRoom: float,
    bathroom: float,
    balcony: str,
    agePossession: str,
    built_up_area: float,
    servant_room: float,
    store_room: float,
    furnishing_type: float,
    luxury_category: str,
    floor_category: str
) -> str:
    
    input_data = pd.DataFrame([[
        property_type, sector.lower(), bedRoom, bathroom, balcony,
        agePossession, built_up_area, servant_room, store_room,
        furnishing_type, luxury_category, floor_category
    ]], columns=[
        'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
        'agePossession', 'built_up_area', 'servant room', 'store room',
        'furnishing_type', 'luxury_category', 'floor_category'
    ])

    prediction = pipeline.predict(input_data)[0]
    return f"₹ {round(prediction, 2)} Crores"


# -------------------------------
# 4. TOOL SCHEMA
# -------------------------------
class PropertyInput(BaseModel):
    property_type: str
    sector: str
    bedRoom: float
    bathroom: float
    balcony: str
    agePossession: str
    built_up_area: float
    servant_room: float
    store_room: float
    furnishing_type: float
    luxury_category: str
    floor_category: str


predict_tool = StructuredTool.from_function(
    func=predict_property_price,
    args_schema=PropertyInput,
    name="predict_property_price",
    description="Predict property price in Gurgaon"
)

# -------------------------------
# 5. LOAD LLM AGENT
# -------------------------------
@st.cache_resource
def load_agent():
    os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"  # 🔑 put your key here

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    tools = [predict_tool]

    return create_react_agent(llm, tools=tools)

agent_executor = load_agent()

# -------------------------------
# 6. UI INPUT
# -------------------------------
st.header("Property Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    property_type = st.selectbox("Property Type", ['flat', 'house'])
    sector = st.text_input("Sector", value='sector 36')
    built_up_area = st.number_input("Built-up Area", min_value=100.0, value=1200.0)
    agePossession = st.selectbox("Age of Property", [
        'New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction'
    ])

with col2:
    bedRoom = st.number_input("Bedrooms", min_value=1.0, value=3.0)
    bathroom = st.number_input("Bathrooms", min_value=1.0, value=2.0)
    balcony = st.selectbox("Balconies", ['0', '1', '2', '3', '3+'])
    floor_category = st.selectbox("Floor Category", ['Low Floor', 'Mid Floor', 'High Floor'])

with col3:
    luxury_category = st.selectbox("Luxury Category", ['Low', 'Medium', 'High'])
    furnishing_type = st.selectbox("Furnishing", [0.0, 1.0, 2.0])
    servant_room = st.selectbox("Servant Room", [0.0, 1.0])
    store_room = st.selectbox("Store Room", [0.0, 1.0])

# -------------------------------
# 7. BUTTON → AGENT CALL
# -------------------------------
if st.button("Generate Valuation", type="primary"):

    query = f"""
    Predict property price with following details:
    property_type: {property_type}
    sector: {sector}
    bedRoom: {bedRoom}
    bathroom: {bathroom}
    balcony: {balcony}
    agePossession: {agePossession}
    built_up_area: {built_up_area}
    servant_room: {servant_room}
    store_room: {store_room}
    furnishing_type: {furnishing_type}
    luxury_category: {luxury_category}
    floor_category: {floor_category}
    """

    try:
        response = agent_executor.invoke({
            "messages": [("user", query)]
        })

        output = response["messages"][-1].content

        st.success("AI Agent completed analysis")
        st.metric(label="Estimated Market Value", value=output)

    except Exception as e:
        st.error(f"Error: {e}")
