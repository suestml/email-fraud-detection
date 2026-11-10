import re
import pandas as pd

def extract_meta_features(data, text_column='full_text'):
    if isinstance(data, pd.DataFrame):
        s = data[text_column].fillna("").astype(str)
    elif isinstance(data, pd.Series):
        s = data.fillna("").astype(str)
    elif isinstance(data, str):
        s = pd.Series([data])
    else:
        s = pd.Series(data).fillna("").astype(str)

    features = pd.DataFrame(index=s.index)

    # 1. Karakter ve kelime sayıları
    features['char_count'] = s.apply(len)
    features['word_count'] = s.apply(lambda x: len(x.split()))

    # 2. Büyük harf oranı
    features['caps_ratio'] = s.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )

    # 3. Ünlem ve Dolar işareti sayıları (Modelin beklediği: dollar_count)
    features['exclamation_count'] = s.apply(lambda x: x.count('!'))
    features['dollar_count'] = s.apply(lambda x: x.count('$'))

    # 4. URL sayısı ve IP içeren link kontrolü (Modelin beklediği: has_ip_url)
    features['url_count'] = s.apply(
        lambda x: len(re.findall(r'https?://\S+|www\.\S+', x, flags=re.IGNORECASE))
    )
    features['has_ip_url'] = s.apply(
        lambda x: int(bool(re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}', x)))
    )

    return features