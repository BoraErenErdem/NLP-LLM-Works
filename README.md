# Doğal Dil İşleme (NLP) — Uyglamalı Öğrenme Rehberi

Bu repo, NLP öğrenme sürecimde tuttuğum kişisel notlar ve kod özetlerinden oluşmaktadır. Temel metin ön işlemeden derin öğrenme modellerine kadar konuları adım adım, Türkçe açıklamalarla ele almaktadır.

> Yeni konular öğrendikçe düzenli olarak güncellenmektedir.

---

## İçerik

```
google_codes/
├── metin_on_isleme/          # Metin temizleme ve hazırlama
├── metin_temsili/            # Sayısal temsil yöntemleri
├── nlp_temel_gorevleri/      # Temel NLP görevleri
└── derin_ogrenme/            # RNN, GRU, LSTM modelleri
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

---

## Kullanılan Kütüphaneler

- **NLTK** — tokenization, stopwords, stemming, lemmatization
- **scikit-learn** — CountVectorizer, TF-IDF, Decision Tree, metrikler
- **TensorFlow / Keras** — RNN, GRU, LSTM, Embedding katmanları
- **gensim** — Word2Vec, FastText
- **spaCy** — NER, morfolojik analiz, POS etiketleme
- **pandas / NumPy** — veri işleme
- **matplotlib** — görselleştirme (PCA)

---

## Kurulum

```bash
pip install nltk scikit-learn tensorflow gensim spacy pandas numpy matplotlib
python -m spacy download en_core_web_sm
```

---

## Öğrenme Yolu

```
Metin Ön İşleme → Metin Temsili → Temel NLP Görevleri → Derin Öğrenme
```

Her modül bir öncekinin üzerine inşa edilmiştir. Sırayla ilerlemek önerilir.
