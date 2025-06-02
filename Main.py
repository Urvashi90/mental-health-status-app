import streamlit as st
import torch
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

model_name = "Urvashi12Dwivedi/mental-health-bert"

model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

st.title("How are you feeling today?")
input_text = st.text_input("Enter your thoughts")

if st.button("detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        predicted_class = detect_anxiety(input_text)
        st.write("Predicted status:", predicted_class)
