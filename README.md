# Spam SMS Classification using NLP

This project implements an SMS spam classification system using Natural Language Processing (NLP) and machine learning. It leverages TF-IDF vectorization to convert messages into features and uses a Random Forest classifier to detect spam.

## 🚀 Live Demo
🔗 Try the deployed Streamlit app here: https://thesmsspamcheck.streamlit.app/

Features:

* Check a single SMS message

* Upload a CSV of messages for batch analysis

* Download the results as a CSV

## 📁 Project Structure

```
spam-sms-classification/
├── data/
│   ├── raw/                # Original SMS dataset
│   └── processed/          # Cleaned and preprocessed data
├── models/                 # Saved models
├── src/
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── feature_engineering.py # Word2Vec and AvgWord2Vec implementation
│   ├── model_training.py      # ML model implementation
│   ├── app.py                 # Streamlit app
│   └── api_service.py         # FastAPI service
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```
## 🧠 Model
Vectorization: TF-IDF

Classifier: Random Forest

Note: Word2Vec + AvgWord2Vec code is present but currently not used in the final pipeline.

## ⚙️ FastAPI
The api_service.py provides a REST API interface but is not used in the Streamlit deployment or locally. No setup instructions are included for FastAPI.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

