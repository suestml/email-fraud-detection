import os
import joblib
import pandas as pd
from scipy.sparse import hstack
from preprocess import extract_meta_features

class EmailFraudPredictor:
    def __init__(self, models_dir: str = "../models", threshold: float = 0.50):
        self.threshold = threshold
        
        # Kaydedilmiş bileşenleri yükle
        self.model = joblib.load(os.path.join(models_dir, "fraud_model.joblib"))
        self.vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        self.scaler = joblib.load(os.path.join(models_dir, "feature_scaler.joblib"))
        
    def predict(self, subject: str, body: str) -> dict:
        full_text = f"{subject} {body}".strip()
        temp_df = pd.DataFrame([{'full_text': full_text}])
        
        # 1. TF-IDF dönüşümü
        text_tfidf = self.vectorizer.transform([full_text])
        
        # 2. Meta özellikleri çıkar ve ölçeklendir
        meta_features = extract_meta_features(temp_df)
        meta_scaled = self.scaler.transform(meta_features)
        
        # 3. Özellikleri birleştir
        combined = hstack([text_tfidf, meta_scaled])
        
        # 4. Tahmin ve olasılık hesabı
        prob = self.model.predict_proba(combined)[0][1]
        is_fraud = bool(prob >= self.threshold)
        
        return {
            'is_fraud': is_fraud,
            'fraud_probability': round(float(prob), 4),
            'risk_level': 'YÜKSEK' if prob >= 0.70 else ('ORTA' if prob >= 0.40 else 'DÜŞÜK'),
            'meta_stats': meta_features.iloc[0].to_dict()
        }

if __name__ == "__main__":
    # Test için örnek e-postalar
    predictor = EmailFraudPredictor(threshold=0.50)
    
    test_emails = [
        {
            "subject": "Project sync meeting notes",
            "body": "Hi team, please find attached the weekly notes and roadmap. See you tomorrow at 10 AM. Best regards."
        },
        {
            "subject": "URGENT: Your bank account is suspended!",
            "body": "Dear customer, unauthorized activity detected. Click immediately: http://192.168.1.15/login-verify and update your PIN!!!"
        }
    ]
    
    for idx, email in enumerate(test_emails, 1):
        result = predictor.predict(email['subject'], email['body'])
        print(f"\n--- ÖRNEK {idx} TAHMİNİ ---")
        print(f"Konu: {email['subject']}")
        print(f"Sonuç: {'DOLANDIRICI (PHISHING)' if result['is_fraud'] else 'NORMAL (TEMİZ)'}")
        print(f"Olasılık: %{result['fraud_probability'] * 100:.2f}")
        print(f"Risk Seviyesi: {result['risk_level']}")