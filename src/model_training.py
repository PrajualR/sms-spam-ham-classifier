import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_preprocessing import load_data, prepare_dataset, download_nltk_resources
# from feature_engineering import word2vec_model, create_avgword2vec, split_dataset
from feature_engineering import create_tfidf_features, split_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    download_nltk_resources()
    # Create directories if they don't exist
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(os.path.join(BASE_DIR, 'data', 'raw', 'SMSSpamCollection.csv'))
    df = prepare_dataset(df)

    # Save preprocessed data
    df.to_csv(os.path.join(f"{BASE_DIR}/data/processed", 'preprocessed_data.csv'), index=False)

    # # Feature engineering
    # print("Creating Word2Vec features...")
    # w2v_model = word2vec_model(df)
    # X = create_avgword2vec(df, w2v_model)
    # y = df['Label'].values

    # Feature extraction using TF-IDF
    print("Extracting TF-IDF features...")
    X, tfidf_vectorizer = create_tfidf_features(df, max_features=1000)
    y = df['Label'].values

    # Split dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(f'{BASE_DIR}/models/confusion_matrix.png')

    # # Save model and Word2Vec model
    # print("Saving models...")
    # with open(f'{BASE_DIR}/models/spam_classifier.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    #
    # w2v_model.save(f'{BASE_DIR}/models/word2vec.model')

    # Save model and vectorizer
    print("Saving model and TF-IDF vectorizer...")
    with open(os.path.join(BASE_DIR, 'models', 'spam_classifier.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # Save label counts for visualization
    label_counts = df['Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    label_counts['Label'] = label_counts['Label'].map({1: 'Spam', 0: 'Ham'})
    label_counts.to_csv(f'{BASE_DIR}/models/label_counts.csv', index=False)

    print(f"Training complete. Models saved in {BASE_DIR}/models/")
    # return model, w2v_model
    return model, tfidf_vectorizer

if __name__ == "__main__":
    train_model()