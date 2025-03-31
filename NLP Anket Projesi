
# 📊 NLP Projesi: Açık Uçlu Anket Geri Bildirimlerini Anlamak

Bu proje, farklı dillerde toplanmış açık uçlu anket yorumlarını işleyerek; **konu analizi (topic classification)** ve **duygu analizi (sentiment analysis)** yapmayı hedefler. Otomatik analiz sayesinde hem zaman kazandırır hem de daha tutarlı sonuçlar üretir.

---

## 🎯 Projenin Amacı

- Serbest metin yorumlarını otomatik olarak analiz etmek
- Yorumun hangi konuya ait olduğunu belirlemek (örn. "kargo", "ürün kalitesi")
- Yorumun duygu durumu nedir? (pozitif / negatif / nötr)
- Tüm analizleri veritabanına kaydedip, görselleştirmek

---

## 🗺️ Aşama Aşama Yol Haritası

---

### ✅ 1. Aşama: Veri Toplama / Yükleme

**Amaç:** Elinde bir `.csv` veya `.xlsx` dosyası olsun. Şu şekilde:

| id | text                            | language |
|----|---------------------------------|----------|
| 1  | "Kargo çok yavaş geldi."       | tr       |
| 2  | "The product quality is great."| en       |

📌 Eğer veri yoksa örnek oluştur:
```python
import pandas as pd

data = {
    "id": [1, 2, 3],
    "text": ["Kargo geç geldi ama paket sağlamdı.", 
             "Customer support was excellent.", 
             "Das Produkt war defekt angekommen."],
    "language": ["tr", "en", "de"]
}

df = pd.DataFrame(data)
df.to_csv("feedback.csv", index=False)
```

---

### ✅ 2. Aşama: Ön İşleme

**Amaç:** Metni temizle, gerekiyorsa dillerini tespit et ve İngilizce’ye çevir.

```python
from deep_translator import GoogleTranslator

def translate_to_en(text, lang):
    if lang == "en":
        return text
    return GoogleTranslator(source=lang, target="en").translate(text)

df["translated"] = df.apply(lambda x: translate_to_en(x["text"], x["language"]), axis=1)
```

---

### ✅ 3. Aşama: Duygu Analizi (Sentiment Analysis)

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

df["sentiment_result"] = df["translated"].apply(lambda x: sentiment(x)[0])
df["sentiment_label"] = df["sentiment_result"].apply(lambda x: x["label"])
```

---

### ✅ 4. Aşama: Konu (Topic) Sınıflandırması

Basit bir örnek:
```python
topics = {
    "delivery": ["shipping", "delivered", "late", "cargo"],
    "product": ["product", "quality", "defect"],
    "support": ["support", "service", "help", "customer"]
}

def detect_topic(text):
    for topic, keywords in topics.items():
        if any(word in text.lower() for word in keywords):
            return topic
    return "other"

df["topic"] = df["translated"].apply(detect_topic)
```

Gelişmiş sistem için `zero-shot-classification` modeli kullanılabilir:
```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["delivery", "product", "support", "payment"]

df["topic_result"] = df["translated"].apply(lambda x: classifier(x, candidate_labels=labels)["labels"][0])
```

---

### ✅ 5. Aşama: Sonuçları Veritabanına veya JSON’a Aktar

```python
df[["id", "translated", "sentiment_label", "topic"]].to_json("results.json", orient="records", lines=True)
```

---

### ✅ 6. Aşama: Dashboard ile Görselleştirme

`Streamlit` ile basit dashboard:
```python
# streamlit_app.py
import streamlit as st
import pandas as pd

df = pd.read_json("results.json", lines=True)

st.title("Anket Geri Bildirim Analizi")
st.write(df)

st.bar_chart(df["sentiment_label"].value_counts())
st.bar_chart(df["topic"].value_counts())
```

Çalıştırmak için:
```bash
streamlit run streamlit_app.py
```

---

## 🚀 Proje Tamamlandığında Neler Elde Edersin?

- Kullanıcı geri bildirimlerinden otomatik analiz
- Hangi konularla ilgili yorumlar daha fazla?
- Genelde pozitif mi yoksa negatif mi konuşulmuş?
- Tüm bunları gerçek zamanlı gösteren bir panel

---

## 🧠 Geliştirme Fikirleri

- Görsel içeren yorumlar için OCR entegrasyonu
- Çok daha detaylı topic modelleri (LDA, BERTopic, etc.)
- Zaman bazlı analiz (haftalık duygu değişimi)
- Excel/Word/PDF içinden veri alma

---

**Hazırlayan:** Gerçek dünyada kullanılabilir, uçtan uca bir NLP proje planı ✨
