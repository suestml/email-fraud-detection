import os
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import extract_meta_features

def load_data(filepath: str) -> pd.DataFrame:
    """Veriyi yükler, temizler ve birleşik metin sütunu üretir."""
    df = pd.read_csv(filepath)
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['full_text'] = df['subject'] + ' ' + df['body']
    
    clean_df = df[df['full_text'].str.strip() != ''].copy()
    
    if clean_df['label'].dtype == 'object':
        clean_df['target'] = clean_df['label'].apply(
            lambda x: 1 if str(x).lower() in ['1', 'phishing', 'spam', 'fraud'] else 0
        )
    else:
        clean_df['target'] = clean_df['label'].astype(int)
        
    return clean_df

def train_pipeline(data_path: str, models_dir: str = "../models"):
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Veriyi hazırla
    print("Veri yükleniyor...")
    df = load_data(data_path)
    
    X = df['full_text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Eğitim: {len(X_train)} satır, Test: {len(X_test)} satır")
    
    # 2. TF-IDF Fit & Transform
    print("TF-IDF özellikleri çıkarılıyor...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 3. Meta Özellikler
    print("Meta özellikler çıkarılıyor...")
    meta_train = extract_meta_features(df.loc[X_train.index])
    meta_test = extract_meta_features(df.loc[X_test.index])
    
    scaler = StandardScaler()
    meta_train_scaled = scaler.fit_transform(meta_train)
    meta_test_scaled = scaler.transform(meta_test)
    
    # 4. Özellikleri birleştir
    X_train_combined = hstack([X_train_tfidf, meta_train_scaled])
    X_test_combined = hstack([X_test_tfidf, meta_test_scaled])
    
    # 5. Model Eğitimi
    print("Model eğitiliyor...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_combined, y_train)
    
    # 6. Test Değerlendirmesi
    y_pred = model.predict(X_test_combined)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- TEST SONUÇLARI ---")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Yanlış Alarm (FP): {cm[0, 1]}")
    print(f"Kaçan Dolandırıcı (FN): {cm[1, 0]}")
    
    # 7. Model ve Vektörleştiricileri Kaydet
    print("\nBileşenler kaydediliyor...")
    joblib.dump(model, os.path.join(models_dir, "fraud_model.joblib"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "feature_scaler.joblib"))
    print(f"Model ve dönüştürücüler başarıyla '{models_dir}/' klasörüne kaydedildi.")

if __name__ == "__main__":
    DATA_PATH = '../data/emails_real.csv'
    train_pipeline(DATA_PATH)