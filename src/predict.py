import os
import joblib
import pandas as pd
from scipy.sparse import hstack
from src.preprocess import extract_meta_features

class FraudPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models")
        
        self.vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        self.scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
        self.model_path = os.path.join(model_dir, "fraud_model.joblib")

        for path in [self.vectorizer_path, self.scaler_path, self.model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model bileşeni bulunamadı: {path}")

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.scaler = joblib.load(self.scaler_path)
        self.model = joblib.load(self.model_path)

    def predict_single(self, subject: str = "", body: str = "", threshold: float = 0.5):
        raw_text = f"{subject} {body}".strip()
        if not raw_text:
            return {"fraud_probability": 0.0, "is_fraud": False}

        # 1. TF-IDF dönüşümü
        text_features = self.vectorizer.transform([raw_text])

        # 2. Meta özellikleri çıkar ve ölçekle ('full_text' sütunlu DataFrame veriyoruz)
        input_df = pd.DataFrame({"full_text": [raw_text]})
        meta_df = extract_meta_features(input_df, text_column="full_text")
        meta_features = self.scaler.transform(meta_df)

        # 3. Özellikleri birleştir
        X_combined = hstack([text_features, meta_features])

        # 4. Tahmin üret
        prob = self.model.predict_proba(X_combined)[0, 1]
        is_fraud = bool(prob >= threshold)

        return {
            "fraud_probability": float(prob),
            "is_fraud": is_fraud,
            "meta_features": meta_df.iloc[0].to_dict()
        }

if __name__ == "__main__":
    predictor = FraudPredictor()
    test_result = predictor.predict_single(
        subject="Urgent Security Alert", 
        body="Please verify your account immediately at http://192.168.1.1"
    )
    print("Test Sonucu:", test_result)