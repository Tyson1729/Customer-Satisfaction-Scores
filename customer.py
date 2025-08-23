import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv(r'C:\Users\sheew\Downloads\customer_feedback_satisfaction.csv')

# Define target and features
X = df.drop(columns=['CustomerID','SatisfactionScore'])
y = df['SatisfactionScore']

# Identify categorical and numerical columns
categorical_cols = ["Gender", "Country", "FeedbackScore", "LoyaltyLevel"]
numerical_cols = ["Age", "Income", "ProductQuality", "ServiceQuality", "PurchaseFrequency"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Define model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor())
])

# Train on full dataset
model.fit(X, y)

# UI
st.header("🔮 Predict Customer Satisfaction")

st.markdown(
    """
    Welcome to the **Customer Satisfaction Prediction App**! 🎉  

    Provide the details below about the customer’s profile, product usage, and experience.  
    Based on your inputs, our model will **predict the satisfaction score (0–100)**.  

    👉 Please fill in the required information carefully to get the most accurate prediction.
    """
)

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=1000, max_value=200000, value=50000)
product_quality = st.slider("Product Quality (1-10)", 1, 10, 5)
service_quality = st.slider("Service Quality (1-10)", 1, 10, 5)
purchase_frequency = st.number_input("Purchase Frequency (last year)", min_value=0, max_value=100, value=5)
gender = st.selectbox("Gender", ["Male", "Female"])
country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "France"])
feedback = st.selectbox("Feedback Score", ["Low", "Medium", "High"])
loyalty = st.selectbox("Loyalty Level", ["Bronze", "Silver", "Gold"])

# Create dataframe for prediction
input_df = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "ProductQuality": [product_quality],
    "ServiceQuality": [service_quality],
    "PurchaseFrequency": [purchase_frequency],
    "Gender": [gender],
    "Country": [country],
    "FeedbackScore": [feedback],
    "LoyaltyLevel": [loyalty]
})

if st.button("Predict Satisfaction Score"):
    prediction = model.predict(input_df)[0]
    prediction = np.clip(prediction, 1, 100)
    st.success(f"Predicted Satisfaction Score: {prediction:.2f}")
