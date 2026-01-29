import tldextract
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# few whitelisted domains (U can use top 100-300 trusted domain through a csv (dont whitelist too much))
# -----------------------------
SAFE_DOMAINS = {
    "youtube.com",
    "google.com",
    "github.com",
    "amazon.com",
    "wikipedia.org",
    "openai.com"
}

MODEL_PATH = "bert-phishing-model"
TOKENIZER_PATH = "bert-phishing-tokenizer"
MAX_LEN = 64
PHISHING_THRESHOLD = 0.6  # confidence threshold

# -----------------------------
# LOAD MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -----------------------------
# HELPERS
# -----------------------------
def get_registered_domain(url: str) -> str:
    """
    Extracts the real registrable domain (anti-bypass).
    """
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}"

def bert_predict(url: str):
    """
    Returns (label, confidence)
    """
    inputs = tokenizer(
        url,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    phishing_prob = probs[0][1].item()
    label = "PHISHING" if phishing_prob >= PHISHING_THRESHOLD else "SAFE"

    return label, phishing_prob

# -----------------------------
# MAIN DECISION FUNCTION
# -----------------------------
def predict_url(url: str):
    url = url.strip()

    # 1️⃣ Extract real domain
    domain = get_registered_domain(url)

    # 2️⃣ Hard whitelist (ONLY real domains)
    if domain in SAFE_DOMAINS:
        return {
            "url": url,
            "domain": domain,
            "prediction": "SAFE",
            "confidence": 1.00,
            "reason": "Trusted registered domain (whitelisted)"
        }

    # 3️⃣ ML prediction
    label, confidence = bert_predict(url)

    return {
        "url": url,
        "domain": domain,
        "prediction": label,
        "confidence": round(confidence, 4),
        "reason": "BERT model inference"
    }

# -----------------------------
# CLI TEST LOOP 
# -----------------------------
if __name__ == "__main__":
    print(" Phishing URL Detector (secure mode)")
    print("Type 'exit' to quit\n")

    while True:
        user_url = input("Enter URL (or exit): ").strip()
        if user_url.lower() == "exit":
            break

        result = predict_url(user_url)

        print("\n🔎 Result:")
        print(f"URL        : {result['url']}")
        print(f"Domain     : {result['domain']}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}")
        print(f"Reason     : {result['reason']}")
        print("-" * 50)
