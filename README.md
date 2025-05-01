# Spam SMS Classification using NLP, Word2Vec and AvgWord2Vec

This project implements an SMS spam classification system using Natural Language Processing (NLP) techniques. It leverages TfidfVectorizer, Word2Vec word embeddings and AvgWord2Vec approach to convert text data into meaningful numerical features for machine learning models.

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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

