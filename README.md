# Doğal Dil İşleme (NLP) — Uyglamalı Öğrenme Rehberi

Bu repo, NLP ve LLM öğrenme sürecimde tuttuğum kişisel notlar ve kod özetlerinden oluşmaktadır. Temel metin ön işlemeden derin öğrenme modellerine ve LLM tabanlı uygulama geliştirmeye kadar konuları adım adım, Türkçe açıklamalarla ele almaktadır.

---

## İçerik

```
google_codes/
├── metin_on_isleme/          # Metin temizleme ve hazırlama
├── metin_temsili/            # Sayısal temsil yöntemleri
├── nlp_temel_gorevleri/      # Temel NLP görevleri
├── derin_ogrenme/            # RNN, GRU, LSTM modelleri
├── gelismis_nlp_gorevleri/   # Transformer tabanlı ileri düzey görevler
├── akilli_asistan/           # Gemini API ile not/etkinlik asistanı (SQLite)
├── doktor_asistani/          # LangChain + Gemini ile doktor asistanı chatbot
├── akilli_turizm_rehberi/    # Ollama (Gemma3) ile turizm rehberi chatbot
├── sozlesme_inceleme_asistani/ # FAISS + Gemini ile RAG tabanlı sözleşme inceleme asistanı
└── musteri_destek_botu/        # LaBSE + FAISS + Ollama (Gemma4) ile RAG tabanlı müşteri destek botu
```

---

## Modüller

### 1. Metin Ön İşleme
| Dosya | Konu |
|-------|------|
| `1_veri_temizleme.py` | Boşluk normalizasyonu, HTML/URL temizleme, büyük/küçük harf |
| `2_tokenizasyon.py` | Kelime ve cümle tokenizasyonu (NLTK) |
| `3_kok_ve_govde_bulma.py` | Stemming (Porter) ve Lemmatization (WordNet) |
| `4_durdurma_kelimeler.py` | Stop words çıkarma (İngilizce, Türkçe, manuel) |

### 2. Metin Temsili
| Dosya | Konu |
|-------|------|
| `1_bow.py` | Bag of Words (CountVectorizer) |
| `2_bow_imdb.py` | BoW — IMDB film yorumları veri seti |
| `3_tf_idf.py` | TF-IDF vektörleştirme |
| `4_tf_idf_sms_spam.py` | TF-IDF — SMS spam tespiti |
| `5_n_grams.py` | Unigram, Bigram, Trigram analizi |
| `6_word_embeddings.py` | Word2Vec ve FastText + PCA görselleştirme |
| `7_word_embeddings_imdb.py` | IMDB ile ileri düzey word embeddings |

### 3. Temel NLP Görevleri
| Dosya | Konu |
|-------|------|
| `1_metin_siniflandirma.py` | SMS spam sınıflandırma (binary classification) |
| `2_varlik_ismi_tanima_(ner).py` | Named Entity Recognition — kişi, yer, kurum (spaCy) |
| `3_morfolojik_analiz.py` | Lemma, POS, çoğul analizi |
| `4_metin_parcasi_etiketleme.py` | Part-of-Speech (POS) etiketleme |
| `5_kelime_anlami_belirsizligi_giderme.py` | Word Sense Disambiguation (Lesk algoritması) |
| `6_duygu_analizi.py` | Sentiment Analysis — VADER ile Amazon yorumları |
| `7_oneri_sistemleri.py` | Neural Collaborative Filtering (embedding tabanlı DNN) |

### 4. Derin Öğrenme
| Dosya | Konu |
|-------|------|
| `1_rnn.py` | RNN ile duygu analizi — restoran yorumları |
| `2_gru.py` | GRU tabanlı sentiment classification — IMDB |
| `3_lstm.py` | LSTM ile metin üretimi (text generation) |

### 5. Gelişmiş NLP Görevleri
| Dosya | Konu |
|-------|------|
| `1_metin_ozetleme.py` | BART (facebook/bart-large-cnn) ile metin özetleme |
| `2_soru_cevap_sistemi.py` | BERT (SQuAD fine-tuned) ile soru cevaplama |
| `3_bilgi_getirme.py` | BERT embedding + cosine similarity ile bilgi getirme |
| `4_metin_cevirisi.py` | MarianMT (Helsinki-NLP) ile İngilizce→Fransızca çeviri |

### 6. Uygulamalı LLM Projeleri

NLP modüllerinden bağımsız, LLM tabanlı uçtan uca uygulama projeleri.

#### Akıllı Asistan (`akilli_asistan/`)
| Dosya | Konu |
|-------|------|
| `assistant.py` | Gemini 2.5 Flash API ile chatbot (ham HTTP istekleri) |
| `database.py` | Notlar ve etkinlikler için SQLite veritabanı işlemleri |
| `main.py` | Proje tanımı ve bileşenlerin birleştirileceği giriş noktası |

#### Doktor Asistanı (`doktor_asistani/`)
| Dosya | Konu |
|-------|------|
| `doktor_asistani_terminal.py` | LangChain + Gemini 2.5 Flash ile terminal tabanlı, hafızalı (memory) doktor asistanı |
| `doktor_asistani_api.py` | FastAPI ile web servisine çevrilmiş doktor asistanı, kullanıcı bazlı memory |
| `client_test.py` | FastAPI servisini test eden istemci scripti |

#### Akıllı Turizm Rehberi (`akilli_turizm_rehberi/`)
| Dosya | Konu |
|-------|------|
| `akilli_turizm_rehberi_terminal.py` | Ollama (Gemma3 4B, local) ile terminal tabanlı turizm rehberi chatbot |
| `akilli_turizm_rehberi_streamlit_streaming.py` | Streamlit arayüzü ve token bazlı streaming yanıt |

#### Sözleşme İnceleme Asistanı (`sozlesme_inceleme_asistani/`)
| Dosya | Konu |
|-------|------|
| `build_vector_db.py` | PDF'den metin çıkarımı (PyMuPDF), chunklama, embedding (sentence-transformers) ve FAISS vektör veritabanı oluşturma |
| `main.py` | Kullanıcı sorusunu vektörleştirip FAISS'te arama yapan ve Gemini 2.5 Flash ile RAG (Retrieval Augmented Generation) tabanlı yanıt üreten soru-cevap sistemi |

#### Müşteri Destek Botu (`musteri_destek_botu/`)
| Dosya | Konu |
|-------|------|
| `load_pdf_and_embedding.py` | FAQ PDF'inden metin çıkarımı (PyPDF), chunk'lama, LaBSE embedding ve FAISS vektör database oluşturma/kaydetme |
| `chatbot_rag_memory.py` | LangChain + Ollama (Gemma4:e4b) ile RAG + ConversationSummaryBufferMemory zinciri kurma ve terminal testi |
| `streamlit_app.py` | Streamlit arayüzü: PDF yükleme, anlık embedding ve RAG + memory + LLM zinciriyle sohbet |

---

## Kullanılan Kütüphaneler

- **NLTK** — tokenization, stopwords, stemming, lemmatization
- **scikit-learn** — CountVectorizer, TF-IDF, Decision Tree, metrikler
- **TensorFlow / Keras** — RNN, GRU, LSTM, Embedding katmanları
- **gensim** — Word2Vec, FastText
- **spaCy** — NER, morfolojik analiz, POS etiketleme
- **pandas / NumPy** — veri işleme
- **matplotlib** — görselleştirme (PCA)
- **Transformers (Hugging Face)** — BART, BERT, MarianMT modelleri
- **PyTorch** — Transformer modellerinin çalışma ortamı
- **LangChain** — LLM zinciri (chain), memory ve prompt yönetimi (doktor asistanı, turizm rehberi, müşteri destek botu)
- **FastAPI / uvicorn** — Doktor asistanını web servisine çevirmek için
- **Streamlit** — Turizm rehberi ve müşteri destek botu için web arayüzü
- **Ollama** — Gemma3 / Gemma4 modellerini local (on-prem) çalıştırmak için
- **Google Gemini API** — Akıllı asistan, doktor asistanı ve sözleşme inceleme asistanı için LLM sağlayıcı
- **SQLite** — Akıllı asistanın not ve etkinlik verilerini saklaması
- **sentence-transformers** — Embedding ile vektörleştirme için (all-MiniLM-L6-v2: sözleşme; LaBSE: müşteri destek botu)
- **FAISS** — Sözleşme inceleme asistanı ve müşteri destek botunda hızlı benzerlik araması yapan vektör veritabanı
- **PyMuPDF** — Sözleşme PDF dosyasından metin çıkarımı için
- **PyPDF** — Müşteri destek botunda FAQ PDF dosyasından metin çıkarımı için

---

## Kurulum

```bash
pip install nltk scikit-learn tensorflow gensim spacy pandas numpy matplotlib
python -m spacy download en_core_web_sm
pip install transformers torch
```

Uygulamalı LLM projelerinin (`akilli_asistan/`, `doktor_asistani/`, `akilli_turizm_rehberi/`, `sozlesme_inceleme_asistani/`, `musteri_destek_botu/`) her biri kendi `requirements.txt` dosyasına sahiptir:

```bash
pip install -r akilli_asistan/requirements.txt
pip install -r doktor_asistani/requirements.txt
pip install -r akilli_turizm_rehberi/requirements.txt
pip install -r sozlesme_inceleme_asistani/requirements.txt
pip install -r musteri_destek_botu/requirements.txt
```

---

## Öğrenme Yolu

```
Metin Ön İşleme → Metin Temsili → Temel NLP Görevleri → Derin Öğrenme → Gelişmiş NLP Görevleri
```

Her modül bir öncekinin üzerine inşa edilmiştir. Sırayla ilerlenilmesi önerilir.

Uygulamalı LLM projeleri (`akilli_asistan`, `doktor_asistani`, `akilli_turizm_rehberi`, `sozlesme_inceleme_asistani`, `musteri_destek_botu`) bu sıralı öğrenme yolundan bağımsız, paralel bir uygulama pratiği olarak yürütülmektedir.
