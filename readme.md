# 🏠 Intelligent Property Price Prediction (Gen AI + Agentic System)

An end-to-end **Machine Learning + Generative AI system** that predicts property prices and provides **intelligent real estate investment advice** using an agent-based architecture.

---

## 🚀 Project Overview

This project goes beyond traditional ML prediction by integrating:

- 📊 Machine Learning (Random Forest) for price prediction  
- 🤖 Agentic AI (LangGraph) for multi-step reasoning  
- 🔍 RAG (FAISS/Chroma) for market insights retrieval  
- 🧠 LLM for decision-making and advisory generation  
- 🌐 Streamlit UI for interactive user experience  

👉 The system not only predicts price but also answers:  
**“Is this property worth buying?”**

---

## 🎯 Problem Statement

Real estate valuation is influenced by multiple dynamic factors such as location, amenities, and market trends.

Traditional ML models:
- Predict prices ✔  
- But **don’t provide reasoning or advice ❌**

This project solves that by combining:
- ML prediction  
- Market context (RAG)  
- AI reasoning (LLM)  

---

## 🧩 Key Features

- 🔮 Accurate price prediction using Random Forest  
- 📊 Feature importance analysis  
- 🤖 AI-powered investment recommendations  
- 📚 Retrieval-Augmented Generation (RAG)  
- 🔄 Multi-step reasoning using LangGraph agents  
- 📄 Structured advisory output (JSON format)  

---

## 📊 Dataset Description

**Source:** Public Housing Dataset (Kaggle)

### Features:
- Location  
- Area (sq ft)  
- Bedrooms  
- Bathrooms  
- Furnishing Status  
- Property Type  

### Target:
- Price (in Crores)

---

## 📈 Exploratory Data Analysis

- Correlation heatmap  
- Distribution plots  
- Outlier detection (IQR)  
- Area vs Price relationship  
- Location-based trends  

### Insights:
- Area has strong positive correlation with price  
- Location significantly impacts valuation  
- Price distribution is right-skewed  

---

## 🧠 Models Implemented

- Linear Regression  
- Random Forest Regressor ✅ (Best)  
- XGBoost Regressor  

### 📌 Final Model Performance:
- **R² Score:** 0.87  
- **MAE:** 0.18 Crores  

---

## 🏗️ System Architecture

![Architecture](<img width="536" height="1902" alt="image" src="https://github.com/user-attachments/assets/ed202516-fc39-43f6-86cd-32bdb01b8b6b" />

)

### Pipeline Flow:
User Input (Streamlit UI)
↓
Data Preprocessing
↓
ML Model (Random Forest)
↓
Predicted Price
↓
LangGraph Agent (State Manager)
↓
RAG (FAISS - Market Data)
↓
LLM Reasoning Engine
↓
Decision Engine
↓
Structured Advisory Report
↓
UI Output

---

## 🤖 Agentic AI Workflow

1. User inputs property details  
2. ML model predicts price  
3. RAG retrieves market insights  
4. LLM compares prediction vs market  
5. Agent generates recommendation  

---

## 📄 Sample Output

```json
{
  "summary": "Property slightly overpriced",
  "comps": "Similar properties are cheaper",
  "action": "Negotiate before buying",
  "risk": "Moderate",
  "disclaimer": "Not financial advice"
}
```

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit  
- **ML:** Scikit-learn, XGBoost  
- **LLM & Agents:** LangChain, LangGraph  
- **Vector DB:** FAISS / Chroma  
- **Data Handling:** Pandas, NumPy  

---

## 💾 Model Serialization

```python
import pickle
pickle.dump(model, open("model.pkl", "wb"))
```

---

## 🚀 Deployment

- Streamlit Cloud  
- Hugging Face Spaces  
- Interactive UI for real-time predictions  

---

## 🔮 Future Improvements

- Real-time market API integration  
- LLM fine-tuning for domain accuracy  
- Multi-city dataset expansion  
- Personalized investment strategies  

---
## 👥 Team

- Nav Prabhat Singh —  ML Pipeline, Documentation(Project Report & README,Agent Workflow Documentation etc) 
- Lakshay Saharan — Demo Video ,Agent Integration   
- Tattva Rajput — Application Deployment & Hosting
- Divyansh — App Development, Model Optimization 

---

## ⭐ Why This Project Stands Out

Unlike traditional ML projects, this system:

✔ Combines ML + Gen AI  
✔ Uses RAG for real-world context  
✔ Performs reasoning, not just prediction  
✔ Produces actionable investment insights  
