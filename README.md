# 🚨 AI PHISHING URL DETECTOR(DevSecOps/AI)

AI-based phishing URL detection using BERT with DevSecOps practices.

## Features
- BERT fine-tuned on phishing URLs
- Streamlit web interface
- Dockerized deployment
- CI pipeline with security scanning
- Reproducible environment

## Run
docker build -t phishing-detector .
docker run -p 8501:8501 phishing-detector

## Security
Static analysis with Bandit
CI automation with GitHub Actions

## NOTE
the original dataset (~96k URLs) is stored locally and not included in this repository.
for faster experimentation,a balanced 10k subset 'Strata'was created and used for training.

### Using Streamlit
streamlit run app.py