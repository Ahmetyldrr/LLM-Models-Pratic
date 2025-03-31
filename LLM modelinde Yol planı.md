
# 🛤️ NLP Tabanlı PDF'ten Soru-Cevap Sistemine Giden Yol Haritası

Bu yol haritası, PDF belgelerden metin çıkararak bu metin üzerinde bir **soru-cevap (QA)** sistemi geliştirmek isteyen geliştiriciler için hazırlanmıştır. Proje, tamamen **Transformer tabanlı** modeller üzerine kuruludur ve kod yazarak öğrenme esas alınır. Aynı zamanda bir web arayüzü ile kullanıma açılması hedeflenmektedir.

---

## 🎯 Nihai Proje Hedefi

📄 **PDF Belgesi** → 🧠 **Metne Dayalı QA Modeli** → ❓ **Soru-Cevap** → 🌐 **Web Arayüzü (API üzerinden)**

---

## 🧱 Aşamalar ve Öğrenme Modülleri

### Aşama 0: Temel Bilgi (1 Hafta)

| Konu | Araç | Kaynak |
|------|------|--------|
| Tokenization, Attention, Transformer Mimarisi | Hugging Face, Colab | 📘 *Hugging Face Transformers Kitabı* Bölüm 2-3 |
| Sequence Classification, QA modeli temelleri | Hugging Face `transformers` | HF Docs + Kitap Bölüm 4 |

---

### Aşama 1: PDF → Metin (1–2 Gün)

| Konu | Araç/Kütüphane |
|------|----------------|
| OCR (gerekirse) | `pytesseract`, `pdf2image`, `PyMuPDF` |
| Doğrudan metin çıkarımı | `pdfplumber`, `PyMuPDF`, `fitz` |
| Sayfa yapısı/taslak analizi | `LayoutLM`, `pdf2json` |

📝 **Çıktı:** PDF dosyasını temiz bir metin dosyasına çeviren kod (Colab)

---

### Aşama 2: QA Modeli Eğitimi veya Kullanımı (1 Hafta)

| Konu | Araç |
|------|------|
| Hazır QA modelleri | `deepset/roberta-base-squad2`, `bert-large-uncased-whole-word-masking-finetuned-squad` |
| RAG (Retrieval-Augmented Generation) | `Haystack`, `LangChain`, `FAISS` |
| Belgeye dayalı QA | `pipeline("question-answering")` kullanımı |

📝 **Çıktı:** Kullanıcı metne soru sorar, model cevap verir.

---

### Aşama 3: Web Arayüz + API Servis (1–2 Hafta)

| Konu | Araç |
|------|------|
| Basit API | `Flask`, `FastAPI` |
| Web arayüzü | `Gradio`, `Streamlit` |
| Model sunumu | `Hugging Face Spaces`, Docker (opsiyonel) |

📝 **Çıktı:** Web sayfası veya Gradio arayüzü üzerinden:  
1. PDF yüklenir  
2. Soru yazılır  
3. Cevap alınır

---

## 📚 O'Reilly Üzerinden Tavsiye Kitaplar

### 🔹 Temel Transformer ve NLP
- **Natural Language Processing with Transformers** (Lewis, Debut, et al.)
  - Hugging Face destekli resmi kitap.
  - Bölüm 2–4 özellikle senin projen için çok uygun.
- **Practical Deep Learning for Coders** (fast.ai)  
  - Derin öğrenmeye gerçekçi projelerle giriş.

### 🔹 OCR, Belgelerle NLP
- **Deep Learning for Document Processing**  
  - Görselden metin çıkarımı, LayoutLM gibi belgelerle ilgili modeller.

### 🔹 Flask, API ve Dağıtım
- **Flask Web Development**  
  - Hafif API geliştirme için ideal başlangıç.
- **Docker for Developers**  
  - Projeyi taşınabilir hale getirme, opsiyonel ama faydalı.

---

## 🔗 Faydalı Linkler

- Hugging Face Transformers Kitabı: [https://transformersbook.com](https://transformersbook.com)
- Hugging Face Model Hub (TR): [https://huggingface.co/models?language=tr](https://huggingface.co/models?language=tr)
- Haystack (QA ve RAG için): [https://haystack.deepset.ai](https://haystack.deepset.ai)
- Hugging Face Docs (Türkçe modeller, pipelines): [https://huggingface.co/docs](https://huggingface.co/docs)

---

## 🧭 Devam İçin Öneri

Her aşama için ayrı `.md` dosyası + Colab defteri oluştur.  
Adımları GitHub’ta repo olarak tut.  
Her deneyim sonunda “öğrendiklerim” şeklinde bir özet yaz.

---

**Hazırlayan:** Sadece senin için, sistemli ve sade bir öğrenme rehberi.


# 🔧 NLP Projesi Yol Haritası – Kod Temelli Başlangıç Rehberi

Bu rehber, PDF'ten metin çıkarma, bu metinle soru-cevap sistemi kurma ve basit bir arayüzle yayına alma sürecinin temel kodlarını içerir.

---

## 📄 Aşama 1: PDF'ten Metin Çıkarma

```python
# pdfplumber ile PDF'ten metin çekme
import pdfplumber

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text

# Kullanım
text = extract_text_from_pdf("ornek.pdf")
print(text[:500])  # ilk 500 karakteri göster
```

Alternatif olarak:
```python
import fitz  # PyMuPDF

def extract_with_fitz(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

text = extract_with_fitz("ornek.pdf")
```

---

## ❓ Aşama 2: Soru-Cevap (QA) Modeli Kullanımı

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = text  # PDF'ten aldığımız metin
question = "Bu belge neyle ilgili?"

result = qa(question=question, context=context)
print("Cevap:", result["answer"])
```

---

## 🌐 Aşama 3: Basit Arayüz (Gradio ile)

```python
import gradio as gr

def answer_question(question, context):
    result = qa(question=question, context=context)
    return result["answer"]

demo = gr.Interface(
    fn=answer_question,
    inputs=["text", "text"],
    outputs="text",
    title="PDF QA Sistemi",
    description="PDF içeriğine dayalı soru-cevap sistemi"
)

demo.launch()
```

---

## 🚀 Ekstra: API ile Yayınlama (Flask Örneği)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/qa", methods=["POST"])
def qa_api():
    data = request.json
    question = data["question"]
    context = data["context"]
    result = qa(question=question, context=context)
    return jsonify({"answer": result["answer"]})

# Çalıştır
# flask --app app.py run
```

---

## 🧠 Notlar

- `pdfplumber` → Metin bazlı PDF'ler için çok başarılıdır.
- `fitz` (`PyMuPDF`) → Görsel destekli belgeler için daha sağlam sonuç verir.
- `deepset/roberta-base-squad2` → İngilizce QA için çok güçlü bir modeldir.
- Türkçe QA için: `savasy/bert-base-turkish-squad` modelini deneyebilirsin.
- Arayüz için `Gradio`, servis için `Flask` başlangıç için yeterlidir.

---

**Hazırlayan:** Projeye başlaman ve korkmadan ilerlemen için mini bir destek rehberi.
