import os
import streamlit as st
import pandas as pd
from src.predict import FraudPredictor
from src.preprocess import extract_meta_features

st.set_page_config(
    page_title="Email Fraud & Phishing Detection",
    page_icon="🛡️",
    layout="wide"
)

# Kenar Çubuğu
with st.sidebar:
    st.header("⚙️ Model Ayarları")
    threshold = st.slider("Karar Eşiği (Threshold)", 0.0, 1.0, 0.50, 0.05)
    st.markdown("---")
    st.markdown("### Proje Mimarisi:")
    st.markdown("- **NLP:** TF-IDF (1-2 N-gram)")
    st.markdown("- **Meta-Features:** 7 Özellik")
    st.markdown("- **Model:** Logistic Regression")

# Başlık
st.markdown("# 🛡️ Email Fraud & Phishing Detection")
st.markdown("E-posta metnini ve konusunu girerek dolandırıcılık riskini ve yapısal özelliklerini analiz edin.")

# Model yükleme
@st.cache_resource
def load_predictor():
    return FraudPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# Giriş Alanları
col_left, col_right = st.columns([1.2, 1])

with col_left:
    subject_input = st.text_input("E-posta Konusu (Subject):", placeholder="Örn: Urgent: Update your account details")
    body_input = st.text_area("E-posta Gövdesi (Body):", placeholder="E-posta içeriğini buraya yapıştırın...", height=200)
    analyze_btn = st.button("🔍 E-postayı Analiz Et", use_container_width=True)

with col_right:
    if analyze_btn:
        raw_text = f"{subject_input} {body_input}".strip()
        
        if not raw_text:
            st.warning("Lütfen analiz için bir konu veya gövde metni girin.")
        else:
            result = predictor.predict_single(subject=subject_input, body=body_input, threshold=threshold)
            prob = result["fraud_probability"]
            is_fraud = result["is_fraud"]

            st.markdown("### 📊 Analiz Sonucu")
            
            if is_fraud:
                st.error(f"🚨 **DİKKAT: DOLANDIRICILIK ŞÜPHESİ!**\n\nRisk Skoru: **%{prob * 100:.2f}**")
            else:
                st.success(f"✅ **GÜVENLİ GÖRÜNÜYOR**\n\nRisk Skoru: **%{prob * 100:.2f}**")

            st.progress(prob)

            # Yapısal İpuçları
            st.markdown("#### 📌 Tespit Edilen Yapısal İpuçları")
            meta_df = extract_meta_features(raw_text)

            m1, m2 = st.columns(2)
            m1.metric("Karakter Sayısı", int(meta_df['char_count'].iloc[0]))
            m2.metric("URL Sayısı", int(meta_df['url_count'].iloc[0]))

            m3, m4 = st.columns(2)
            m3.metric("Dolar İşareti ($)", int(meta_df['dollar_count'].iloc[0]))
            has_ip = "Evet" if int(meta_df['has_ip_url'].iloc[0]) == 1 else "Hayır"
            m4.metric("IP Linki Var mı?", has_ip)
    else:
        st.info("Sol taraftaki kutulara metin girip **'E-postayı Analiz Et'** butonuna basın.")