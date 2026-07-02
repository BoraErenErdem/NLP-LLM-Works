# Multi-Tool AI Agent — Gemini 2.5 Flash + LangChain

Langchain, Llama 3.3:70b-versatile ve Google Gemini 2.5 Flash tabanlı, beş farklı araca sahip, belleği olan, FastAPI üzerinden servis eden ve Streamlit ile kullanıcı arayüzü sunan çok araçlı yapay zeka ajan sistemi.

---

## Proje Mimarisi

```
Kullanıcı (Streamlit UI)
        │
        ▼
FastAPI  /ask  endpoint
        │
        ▼
LangChain Agent  (ReAct — ZERO_SHOT_REACT_DESCRIPTION)
        │
  ┌─────┴──────────────────────────────────────┐
  │                                            │
  ▼                                            ▼
Tool Seçimi                              Memory
(RAG / Calculator / Discount /     ConversationSummaryBufferMemory
 WebSearch / CurrencyConverter)          (kullanıcı bazlı)
  │
  ▼
Gemini 2.5 Flash/Llama 3.3:70b-versatile  →  Yanıt Üretimi
```

---

## Özellikler

| Özellik | Detay |
|---|---|
| LLM Modelleri | Google Gemini 2.5 Flash · Groq Llama 3.3 70b Versatile |
| Ajan Türü | ReAct (ZERO_SHOT_REACT_DESCRIPTION) |
| Bellek | ConversationSummaryBufferMemory (kullanıcı bazlı) |
| Vektör Veritabanı | FAISS (CPU) |
| Embedding Modeli | LaBSE — çok dilli, Türkçe destekli |
| RAG Belgesi | müşteri destek SSS (PDF) |
| Web Araması | SerpAPI |
| Döviz Kuru | exchangerate-api.com (gerçek zamanlı) |
| API | FastAPI + Uvicorn |
| Arayüz | Streamlit |

---

## Araçlar (Tools)

### 1. RAG Tool
Müşteri destek SSS PDF belgesini FAISS vektör veritabanına aktarır. Kullanıcı sorusu LaBSE embedding modeli ile vektörleştirilir, en alakalı 3 chunk geri çağrılır ve Gemini'ye bağlam olarak sunulur.

```
Örnek: "Ürün iade koşulları nelerdir?"
```

### 2. Calculator Tool
Python `eval()` ile matematiksel ifadeleri değerlendirir. ReAct döngüsünün sonucu kolayca ayrıştırabilmesi için `answer: <sonuç>` formatında döndürür.

```
Örnek: "15 * 10 + 33"  →  "answer: 183"
```

### 3. Discount Tool
Doğal dil girdisinden fiyat bilgisini regex ile çıkarır ve %10 indirim uygular.

```
Örnek: "Telefon fiyatı 1000 TL"  →  "discounted price: 900.00 tl"
```

### 4. Web Search Tool (SerpAPI)
`SerpAPIWrapper` ile Google araması yapar ve gerçek zamanlı web sonuçlarını getirir.

```
Örnek: "Bugün Alanya'da hava kaç derece?"
```

### 5. Currency Converter Tool
`exchangerate-api.com` üzerinden güncel döviz kurunu çeker ve çeviri yapar.

```
Örnek: "5900 USD to TRY"  →  "answer: 5900 USD = 190432.50 TRY"
```

---

## Proje Yapısı

```
ai_agents_yapay_zeka_asistani/
├── main_agent.py               # Temel ajan — terminal üzerinden prototip testi
├── fast_api.py                 # FastAPI /ask endpoint — kullanıcı bazlı memory
├── app_streamlit.py            # Streamlit sohbet arayüzü
├── client.py                   # Terminal istemcisi — FastAPI'ye istek gönderir
├── requirements.txt            # Bağımlılıklar
├── .env                        # API anahtarları (git'e eklenmez)
├── data/
│   └── musteri_destek_faq.pdf  # RAG için müşteri destek belgesi
└── tools/
    ├── calculator_tool.py      # Matematiksel hesaplayıcı
    ├── custom_discount_tool.py # İndirim hesaplayıcı
    ├── rag_tool.py             # FAISS + LaBSE tabanlı RAG
    └── currency_converter_tool.py  # Döviz çevirici
```

---

## Kurulum

### Gereksinimler

- Python 3.10+
- Google Gemini API anahtarı
- Groq API anahtarı
- SerpAPI anahtarı

### Adımlar

```bash
# Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Bağımlılıkları yükle
pip install langchain langchain-community langchain-classic langchain-google-genai
pip install langchain-groq google-generativeai faiss-cpu sentence-transformers
pip install fastapi uvicorn streamlit serpapi google-search-results
pip install pypdf python-dotenv requests
```

### Ortam Değişkenleri

`.env` dosyası oluştur:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
SERPAPI_API_KEY=your_serpapi_key
```

---

## Çalıştırma

### 1. FastAPI Sunucusunu Başlat

```bash
uvicorn fast_api:app --reload
```

API `http://127.0.0.1:8000` adresinde çalışır.  
Swagger dokümantasyonu: `http://127.0.0.1:8000/docs`

### 2. Streamlit Arayüzünü Başlat

```bash
streamlit run app_streamlit.py
```

### 3. Terminal İstemcisi (Opsiyonel)

FastAPI çalışırken terminal üzerinden test etmek için:

```bash
python client.py
```

### 4. Direkt Ajan (Prototip)

FastAPI olmadan direkt terminal'de test:

```bash
python main_agent.py
```

---

## Kullanım Örnekleri

| Senaryo | Örnek Sorgu |
|---|---|
| Karşılama | "Merhaba, sen kimsin?" |
| Matematik | "Bana 15 × 10 + 33 sorusunun cevabını verebilir misin?" |
| İndirim | "Telefonun fiyatı 1000 TL, buna indirim uygula." |
| Hava Durumu | "Bugün Alanya'da hava kaç derece?" |
| Müşteri Desteği | "Bir ürün aldım, geri iade edebilir miyim?" |
| Bellek | "Şimdiye kadar seninle ne konuştuk?" |
| Döviz | "Elimde 5900 USD var, bunu TRY'ye çevirebilir misin?" |

---

## Sistem Akışı

```
Kullanıcı sorusu
      │
      ▼
FastAPI /ask endpoint
      │  (user_id + message)
      ▼
Memory'den geçmiş konuşmayı çek
      │
      ▼
Gemini 2.5 Flash — ReAct döngüsü başlar
      │
      ├── Düşün (Thought)
      ├── Araç seç (Action)
      ├── Aracı çağır (Action Input)
      ├── Sonucu al (Observation)
      └── Yanıtı üret (Final Answer)
      │
      ▼
Memory güncelle (kullanıcı + ai mesajı)
      │
      ▼
JSON yanıt → Streamlit UI
```

---

## Teknoloji Yığını

```
LLM           →  Google Gemini 2.5 Flash  |  Groq Llama 3.3 70b
Agent         →  LangChain ZERO_SHOT_REACT_DESCRIPTION
Memory        →  ConversationSummaryBufferMemory
Embedding     →  sentence-transformers/LaBSE (çok dilli)
Vector DB     →  FAISS (CPU)
PDF Yükleme   →  PyPDFLoader
Web Arama     →  SerpAPI
Döviz         →  exchangerate-api.com
API           →  FastAPI + Uvicorn
UI            →  Streamlit
Env Yönetimi  →  python-dotenv
```

---

## Sonuç

Bu proje üretken yapay zeka ajanlarının nasıl bir şekilde düşünebilen sistemler haline geldiğini gösteren projedir. Langchain ve Gemini entegrasyonu sayesinde multi-tool, memory, rag altyapılı, etkileşimli fastapi ve streamlit arayüz destekli, mlops dev ops (docker-deployment) ile bir akıllı sistem ortaya çıkarılmış olur.
