import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

# -----------------------
# Load Model & Tokenizer
# -----------------------
MODEL_PATH = r"C:\Users\sheew\OneDrive\Desktop\test\model"  # absolute path

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Label mapping (must match training order)
label_mapping = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Irrelevant"}

# -----------------------
# Streamlit UI
# -----------------------
st.title("💬 Sentiment Classifier")
st.write("Enter a comment and the model will classify it into **Positive, Negative, Neutral, or Irrelevant**.")

# Input box
user_input = st.text_area("Your Comment:", "")

if st.button("Classify"):
    if user_input.strip():
        # Tokenize input
        inputs = tokenizer(
            user_input,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Run prediction
        outputs = model(inputs)
        logits = outputs.logits
        predicted_class = tf.argmax(logits, axis=1).numpy()[0]

        # Show result
        st.subheader("Prediction:")
        st.success(f"**{label_mapping[predicted_class]}**")

    else:
        st.warning("⚠️ Please enter a comment before classifying.")