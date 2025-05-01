from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import nltk
import os
# from gensim.models import Word2Vec
# from nltk.tokenize import word_tokenize
import uvicorn

from data_preprocessing import preprocess_text, download_nltk_resources
# from feature_engineering import get_avg_word2vec

download_nltk_resources()

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create FastAPI app
app = FastAPI(
    title="SMS Spam Classifier API",
    description="API for classifying SMS messages as spam or ham",
    version="1.0.0",
)


# Define input model
class Message(BaseModel):
    message: str

    class Config:
        schema_extra = {
            "example": {
                "message": "Congratulations! You've won a free gift card. Click here to claim now!"
            }
        }


# Define response model
class PredictionResponse(BaseModel):
    is_spam: bool
    prediction: str
    probability_spam: float
    probability_ham: float


# Global variables for models
model = None
# w2v_model = None
tfidf_vectorizer = None

@app.on_event("startup")
def load_models():
    """Load models on startup"""
    # global model, w2v_model
    global model, tfidf_vectorizer

    try:
        # Go up one level from /src to project root
        base_path = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_path, ".."))

        # Correct model paths
        model_path = os.path.join(project_root, "models", "spam_classifier.pkl")
        # w2v_path = os.path.join(project_root, "models", "word2vec.model")
        tfidf_path = os.path.join(project_root, "models", "tfidf_vectorizer.pkl")

        # Load models
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # w2v_model = Word2Vec.load(tfidf_path)
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Make sure to run model_training.py first!")

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "SMS Spam Classifier API is running",
        "endpoints": {
            "POST /predict": "Predict if a message is spam or ham"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_spam(message_data: Message):
    """Predict if a message is spam or ham"""
    # global model, w2v_model
    global model, tfidf_vectorizer

    # if model is None or w2v_model is None:
    if model is None or tfidf_vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Models not loaded. Run model_training.py first."
        )

    try:
        # Preprocess the text
        cleaned_text = preprocess_text(message_data.message)

        # Tokenize the cleaned text
        # tokens = word_tokenize(cleaned_text)

        # Convert to word embeddings
        # vector = get_avg_word2vec(tokens, w2v_model, w2v_model.vector_size)
        vector = tfidf_vectorizer.transform([cleaned_text])

        # Reshape for model input (single sample)
        # vector = vector.reshape(1, -1)

        # Make prediction
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        return {
            "is_spam": bool(prediction),
            "prediction": "SPAM" if prediction == 1 else "HAM (Not Spam)",
            "probability_spam": float(probabilities[1]),
            "probability_ham": float(probabilities[0])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing prediction: {str(e)}"
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    # global model, w2v_model
    global model, tfidf_vectorizer

    # if model is None or w2v_model is None:
    if model is None or tfidf_vectorizer is None:
        return {"status": "error", "message": "Models not loaded"}

    return {"status": "ok", "message": "Service is healthy"}


if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)