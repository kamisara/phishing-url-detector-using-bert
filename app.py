import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(
    "./bert-phishing-tokenizer",
    local_files_only=True
)

model = BertForSequenceClassification.from_pretrained(
    "./bert-phishing-model",
    local_files_only=True
)

model.to(device)
model.eval()

def predict_url(url):
    url = url.strip().split("?")[0]
    enc = tokenizer(
        [url],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(
            enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device)
        )
        pred = torch.argmax(out.logits, dim=1).item()
    return "Safe ✅" if pred == 0 else "Phishing !!"
st.title("🚨 AI Phishing URL Detector")
st.write("BERT + DevSecOps mini-project ")
url_input = st.text_input("Enter a URL")
if st.button("Check URL"):
    if url_input.strip() == "":
        st.warning("Enter a URL first ")
    else:
        st.success(f"Prediction: {predict_url(url_input)}")
