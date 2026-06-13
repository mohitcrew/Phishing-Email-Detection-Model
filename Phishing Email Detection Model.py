import argparse
import os
import re
import sys
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


def clean_text(text: str) -> str:
    """Basic text cleaning for email bodies: lowercase, replace URLs and emails, remove non-letters/numbers."""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].apply(clean_text)
    return df


def build_pipeline(max_features: int = 5000) -> Pipeline:
    vect = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1, 2))
    clf = MultinomialNB()
    return Pipeline([('tfidf', vect), ('nb', clf)])


def train_and_evaluate(csv_path: str, model_path: str = None, test_size: float = 0.2, random_state: int = 42) -> Pipeline:
    df = load_dataset(csv_path)
    X = df['text']
    y = df['label']

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if model_path:
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")

    return pipeline


def load_model(path: str) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_text(pipeline: Pipeline, text: str) -> Tuple[str, float]:
    txt = clean_text(text)
    pred = pipeline.predict([txt])[0]
    prob = None
    if hasattr(pipeline, 'predict_proba') or hasattr(pipeline.named_steps['nb'], 'predict_proba'):
        try:
            proba = pipeline.predict_proba([txt])[0]
            # return probability for predicted class
            idx = list(pipeline.classes_).index(pred)
            prob = proba[idx]
        except Exception:
            prob = None
    return pred, prob


def main():
    parser = argparse.ArgumentParser(description='Train or use a phishing email detection model')
    parser.add_argument('-t', '--train', help='Path to training CSV (contains text,label)')
    parser.add_argument('-m', '--model', default='phishing_model.joblib', help='Path to save/load model')
    parser.add_argument('-p', '--predict', help='Predict a single email string')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split proportion when training')
    args = parser.parse_args()

    pipeline = None
    if args.train:
        try:
            pipeline = train_and_evaluate(args.train, model_path=args.model, test_size=args.test_size)
        except Exception as e:
            print('Training error:', e)
            sys.exit(1)
    else:
        # Try loading existing model
        try:
            pipeline = load_model(args.model)
            print(f'Loaded model from {args.model}')
        except Exception:
            print('No trained model found. Use --train to train a model first.')
            if not args.predict:
                sys.exit(1)

    if args.predict:
        if pipeline is None:
            print('Model not available for prediction.')
            sys.exit(1)
        pred, prob = predict_text(pipeline, args.predict)
        if prob is not None:
            print(f'Prediction: {pred} (probability: {prob:.2f})')
        else:
            print(f'Prediction: {pred}')


if __name__ == '__main__':
    main()