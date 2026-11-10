# End-to-End Email Fraud & Phishing Detection Pipeline

Bu proje, e-posta metinleri ve biçimsel meta-özellikler üzerinden oltalama (phishing) ve dolandırıcılık e-postalarını tespit eden uçtan uca makine öğrenmesi boru hattıdır.

## 🚀 Proje Özeti & Metodoloji

* **Veri Seti:** ~39.000 satırlık çok kaynaklı gerçek dünya e-posta veri seti (`naserabdullahalam/phishing-email-dataset`).
* **Temel Model:** TF-IDF n-gram (1,2) + Lojistik Regresyon (Baseline Accuracy: %99.41, ROC-AUC: 0.9996).
* **Hata Analizi (Error Analysis):** Modelin yanıldığı False Positive (normal e-postaya spam denmesi) ve False Negative (kaçırılan dolandırıcılıklar) vakaları incelenmiş; saldırganların sahte antivirüs damgaları ve IP içeren linkler kullandığı tespit edilmiştir.
* **Özellik Mühendisliği (Feature Engineering):** Metin tabanlı TF-IDF vektörlerine ek olarak; URL sayısı, IP adresi varlığı, büyük harf oranı, aciliyet bildiren semboller (`!`, `$`) ve karakter uzunluğu gibi 7 adet yapısal özellik çıkarılmıştır.
* **Sonuç:** Yapısal özelliklerin entegrasyonuyla test setindeki toplam hata sayısı 46'dan 39'a düşürülmüş (FP: 21, FN: 18), modelin dil manipülasyonlarına karşı gürbüzlüğü artırılmıştır.

## 📁 Proje Yapısı

```text
├── data/               # Ham veri seti (Git'e eklenmez)
├── models/             # Eğitilmiş modeller ve scaler/vectorizer bileşenleri
├── notebooks/          # Keşifçi veri analizi ve hata analizi (01_baseline_model.ipynb)
├── src/                # Üretime hazır modüler kodlar
│   ├── preprocess.py   # Meta-özellik çıkarım fonksiyonları
│   ├── train.py        # Pipeline eğitimi ve model serileştirme (joblib)
│   └── predict.py      # Canlı tahmin sınıfı (Inference engine)
├── .gitignore
├── README.md
└── requirements.txt