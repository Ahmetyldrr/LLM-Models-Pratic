
# 🚀 Kendi Verinle Transformer Model Eğitimi: Öğrenme Rehberi

Bu rehber, **kendi verinle, kendi etiketlerinle** bir Transformer (örnek: BERT) modelini nasıl eğiteceğini adım adım açıklar. Öğrenme sürecini hızlandırmak ve sorularını cevaplamak için hazırlanmıştır.

---

## 🎯 Amaç

> **Kendi metin verim** + **kendi etiketlerim** = **özelleştirilmiş sınıflandırma modeli**

Örneğin:
- Duygu analizi (pozitif / negatif / nötr)
- Konu sınıflandırma (spor / politika / magazin)
- Şikayet tespiti (ürün / teslimat / hizmet)

---

## 🔹 Veri Yapısı Nasıl Olmalı?

Verin `.csv` dosyasında şu şekilde olabilir:

| text                                | label     |
|-------------------------------------|-----------|
| “Kargo geç geldi ama ürün güzeldi”  | teslimat  |
| “Telefon 1 hafta içinde bozuldu.”   | ürün      |
| “Müşteri hizmetleri çok iyiydi.”    | hizmet    |

Etiketler senin seçimine göre değişebilir.

---

## 🔸 1. Gerekli Kütüphaneler

```bash
pip install transformers datasets pandas scikit-learn
```

---

## 🔸 2. CSV Verisini Yükle

```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("verin.csv")  # kendi dosya adını koy
dataset = Dataset.from_pandas(df)
```

---

## 🔸 3. Tokenizer ve Model Kurulumu

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_ckpt = "bert-base-uncased"  # Türkçe ise: dbmdz/bert-base-turkish-cased
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

labels = list(set(df['label']))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length')

dataset = dataset.map(tokenize)
dataset = dataset.rename_column("label", "labels")
dataset = dataset.class_encode_column("labels")

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
```

---

## 🔸 4. Eğitim Ayarları

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./model-out",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="no",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()
```

---

## 🔸 5. Tahmin Yapmak

```python
text = "Telefonun şarjı çok çabuk bitiyor."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_id = outputs.logits.argmax(dim=-1).item()
print("Tahmin:", id2label[predicted_id])
```

---

## ❓ Sık Sorular & Cevaplar

### 📌 Verim Türkçe, ne yapmalıyım?
→ Model olarak `dbmdz/bert-base-turkish-cased` kullan.

### 📌 CSV’yi nasıl oluşturmalıyım?
→ `text` ve `label` başlıklı 2 sütun yeterlidir.

### 📌 Etiket sayısı 10’dan fazlaysa?
→ `num_labels` otomatik ayarlanır, sınır yok.

### 📌 Küçük model istiyorum?
→ `distilbert-base-uncased` gibi modeller kullanabilirsin.

---

## 🔗 Faydalı Linkler

- [Hugging Face Turkish Models](https://huggingface.co/models?language=tr&search=bert)
- [Datasets Belgeleri](https://huggingface.co/docs/datasets/)
- [Transformers Belgeleri](https://huggingface.co/docs/transformers/)
- [YouTube: Türkçe BERT Eğitimi](https://www.youtube.com/watch?v=3M8xL3xKJ2I)

---

**Hazırlayan:** Santiago Delgado (Santi)  
**Amacı:** Öğrenmeni hızlandırmak ve sana özel proje oluşturmanı kolaylaştırmak ✨
