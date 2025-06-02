import streamlit as st
import torch
import pickle 
import re
import nltk
from nltk.corpus import stopwords
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
import pandas as pd

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# Load model, tokenizer, label encoder
model = AutoModelForSequenceClassification.from_pretrained("Urvashi12Dwivedi/mental-health-bert")
tokenizer = AutoTokenizer.from_pretrained("Urvashi12Dwivedi/mental-health-bert")
label_encoder = pickle.load(open("label_encoder.pkl","rb"))

stop_words = set(stopwords.words('english'))

def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)  # remove punctuation
    statement = re.sub(r'\d+', '', statement)      # remove numbers
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    cleaned_statement = ' '.join(words)
    return cleaned_statement

def detect_anxiety_with_confidence(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()
    label = label_encoder.inverse_transform([predicted_class])[0]
    return label, confidence

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.title("How are you feeling today?")
st.write("We are here to listen to everything you want to share...")

input_text = st.text_area("Enter your thoughts", placeholder="We are here to listen to everything you want to share...")

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing your input...'):
            predicted_class, confidence = detect_anxiety_with_confidence(input_text)
            sentiment = analyze_sentiment(input_text)
            st.success(f"Predicted status: **{predicted_class}** (Confidence: {confidence:.2f})")
            st.info(f"Sentiment of input: **{sentiment}**")
            # Save history
            st.session_state.history.append((input_text, predicted_class, confidence, sentiment))

# Show history if available
if st.session_state.history:
    st.write("### Your past inputs and predictions:")
    for i, (text, pred, conf, sent) in enumerate(st.session_state.history):
        st.write(f"**Input {i+1}:** {text}")
        st.write(f"**Prediction:** {pred} (Confidence: {conf:.2f})")
        st.write(f"**Sentiment:** {sent}")
        st.write("---")

# Download option for history
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history, columns=["Input Text", "Prediction", "Confidence", "Sentiment"])
    csv = df.to_csv(index=False)
    st.download_button(label="Download Results as CSV", data=csv, file_name="mental_health_results.csv", mime="text/csv")

# Optional: Reset button
if st.button("Reset"):
    st.session_state.history = []
    st.experimental_rerun()
