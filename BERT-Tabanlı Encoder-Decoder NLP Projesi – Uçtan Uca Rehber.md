
# 🚀 BERT-Tabanlı Encoder-Decoder NLP Projesi – Uçtan Uca Rehber

Bu projede, BERT tabanlı bir encoder-decoder mimarisi kullanılarak bir **metin özetleme** sistemi kurulacaktır. Proje; veri hazırlama, model eğitimi, değerlendirme ve dağıtımı (deployment) adımlarını içerir.

---

## 📁 1. Örnek Veri Kümesi

Proje için kullanabileceğimiz örnek veri kümesi **CNN/DailyMail dataset** olabilir. Bu veri seti haber metinlerini ve onların kısa özetlerini içerir.

### 📊 Örnek Format:

| Article | Summary |
|---------|---------|
| "The stock market crashed today due to..." | "Stock market crash." |
| "Scientists discovered a new planet..." | "New planet found." |

Alternatif olarak basit bir örnek CSV oluşturabilirsin:

```python
import pandas as pd

data = {
    "article": [
        "The Eiffel Tower is located in Paris and was built in 1889.",
        "The Amazon rainforest is home to diverse wildlife and vegetation.",
    ],
    "summary": [
        "Eiffel Tower is in Paris.",
        "Amazon rainforest has diverse wildlife."
    ]
}

df = pd.DataFrame(data)
df.to_csv("summary_dataset.csv", index=False)
```

---

## ⚙️ 2. Proje Ortamı ve Gereksinimler

```bash
pip install transformers datasets torch gradio
```

---

## 🧠 3. Model Eğitimi (Training)

Burada `encoder-decoder` modeli olarak Hugging Face'ten `bert2bert` kullanılabilir:

```python
from transformers import EncoderDecoderModel, BertTokenizer, Trainer, TrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset
import torch

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Veri Yükle
import pandas as pd
dataset = pd.read_csv("summary_dataset.csv")

# Tokenization fonksiyonu
def preprocess_function(examples):
    inputs = tokenizer(examples["article"], padding="max_length", truncation=True, max_length=256)
    targets = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Dataset formatı
ds = Dataset.from_pandas(dataset)
tokenized_ds = ds.map(preprocess_function, batched=True)

# Model yükle
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# Training args
training_args = TrainingArguments(
    output_dir="./bert2bert-summary",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    evaluation_strategy="no",
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer
)

trainer.train()
```

---

## ✅ 4. Model Testi (Inference)

```python
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    output = model.generate(**inputs, max_length=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_summary("The Amazon rainforest is home to many species..."))
```

---

## 🌐 5. Deployment (Gradio ile Web Arayüz)

```python
import gradio as gr

interface = gr.Interface(
    fn=generate_summary,
    inputs="text",
    outputs="text",
    title="BERT2BERT Summarizer",
    description="Makale girişinden özet üretir."
)

interface.launch()
```

---

## 🛠️ 6. Alternatif Deployment Yöntemleri

| Yöntem | Açıklama |
|--------|----------|
| `Flask` / `FastAPI` | API olarak sunmak için ideal |
| `Docker` | Her yerde aynı ortamda çalıştırmak için |
| `Hugging Face Spaces` | Gradio + Web sunumunu barındırmak için |
| `Streamlit` | Görselleştirme de gerekiyorsa kullanılabilir |

---

## 🔚 Sonuç

Bu projede:
- BERT tabanlı encoder-decoder kullanarak özetleme modeli eğitildi
- Basit bir veri seti ile çalışıldı
- Model Gradio ile servis edildi

Gerçek veri setleri ile performans artışı sağlanabilir. Bu yapıyı soru-cevap, çeviri gibi seq2seq görevlerde de kullanabilirsin.

