import streamlit as st
import torch
import pickle 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import nltk
from nltk.corpus import stopwords
#load save model


#Custom function


# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

model = AutoModelForSequenceClassification.from_pretrained("C:/Users/HP/Downloads/saved_mental_status_bert")
tokenizer = AutoTokenizer.from_pretrained("C:/Users/HP/Downloads/saved_mental_status_bert")
label_encoder = pickle.load(open("label_encoder.pkl","rb"))

# Get English stopwords from NLTK
stop_words = set(stopwords.words('english'))

def clean_statement(statement):
    # Convert to lowercase
    statement = statement.lower()

    # Remove special characters (punctuation, non-alphabetic characters)
    statement = re.sub(r'[^\w\s]', '', statement)

    # Remove numbers (optional, depending on your use case)
    statement = re.sub(r'\d+', '', statement)# Tokenize the statement (split into words)
    words = statement.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Rejoin words into a cleaned statement
    cleaned_statement = ' '.join(words)

    return cleaned_statement



#
def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]
#UI

st.title("Mental Health status detection bert")

input_text= st.text_input("Enter your thoughts")

if st.button("detect"):
    predicted_class = detect_anxiety(input_text)
    st.write("Predicted status:", predicted_class)
