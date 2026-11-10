import re
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_meta_features(df_subset: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
    """E-posta metinlerinden kural tabanlı ve yapısal sayısal özellikleri çıkarır."""
    features = pd.DataFrame(index=df_subset.index)
    
    # Karakter ve kelime sayıları
    features['char_count'] = df_subset[text_column].apply(len)
    features['word_count'] = df_subset[text_column].apply(lambda x: len(x.split()))
    
    # Büyük harf oranı
    features['caps_ratio'] = df_subset[text_column].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    
    # Sembol sıklıkları
    features['exclamation_count'] = df_subset[text_column].apply(lambda x: x.count('!'))
    features['dollar_count'] = df_subset[text_column].apply(lambda x: x.count('$'))
    
    # Link sayıları ve IP tespiti
    features['url_count'] = df_subset[text_column].apply(
        lambda x: len(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', x))
    )
    features['has_ip_url'] = df_subset[text_column].apply(
        lambda x: 1 if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', x) else 0
    )
    
    return features


if __name__ == "__main__":
    # Küçük bir sahte veriyle fonksiyonu test edelim
    sample_df = pd.DataFrame({
        'full_text': [
            'Hello team, meeting at 10 AM.',
            'URGENT! Claim your $1000 prize now at http://192.168.1.1/win !!!'
        ]
    })
    
    output = extract_meta_features(sample_df)
    print("Preprocess modülü başarıyla çalıştı! Örnek çıktı:")
    print(output)