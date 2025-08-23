import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Load training dataset
df = pd.read_csv('customer_feedback_satisfaction.csv')

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

# Train model on full dataset
model.fit(X, y)

# Streamlit UI
st.header("🔍 Let's Analyze Your Customer Base! ✨")

st.markdown(
    """
    👋 Welcome, Data Explorer!  

    Ever wondered **how happy your customers really are**?  
    We’ve got your back — upload your data, sit back, and let us reveal the story hidden in the numbers. 📊  

    Think of it as giving your customers a voice without sending out another boring survey 😉  
    """
)

uploaded_file = st.file_uploader("📂 Ready to dive in? Drop your customer CSV file here and let’s uncover the secrets!", type=["csv"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)

    # Drop CustomerID if exists in uploaded file
    if "CustomerID" in input_data.columns:
        input_data = input_data.drop(columns=["CustomerID"])

    # Ensure all required columns are present
    missing_cols = [col for col in (numerical_cols + categorical_cols) if col not in input_data.columns]
    if missing_cols:
        st.error(f"⚠️ Oops! Looks like we’re missing a few puzzle pieces: {missing_cols}\n\nHelp us complete the picture!")
    else:
        # Predict satisfaction score
        predictions = model.predict(input_data)
        predictions = np.clip(predictions, 1, 100)
        input_data["PredictedSatisfactionScore"] = predictions

        # Show preview
        st.success("🎉 Predictions are ready — let’s see how your customers feel 💖")
        st.dataframe(input_data)

        # Download option
        csv = input_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Grab Your Insights as CSV",
            data=csv,
            file_name="predicted_satisfaction_scores.csv",
            mime="text/csv"
        )
