
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

