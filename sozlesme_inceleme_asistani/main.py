

"""
Problem tanımı: Sözleşme İnceleme Asistanı
    - kullanıcının yüklediği sözleşme dökümanından bilgi çıkarımı yap.
    - içeriği vektörel olarak temsil et yani embedding yap..!
    - faiss ile hızlı arama yapabilen vektör veri tabanı oluştur.
    - kullanıcıdan soruları al, database'e git ve bilgiyi getir, bu bilgi ve kullanıcı sorusu doğrultusunda gemini cevap üretir.

Kullanılan teknolojiler:
    - embedding -> metni vektörel hale getirme yani sayısallaştırma
    - faiss -> hızlı benzerlik araması için veri tabanı
    - gemini -> bulut ortamında çalışan llm (gemini 2.5 flash)

RAG (Retrieval Augmented Generation) -> Dil modellerine bilgi desteği sağlayan tekniktir.
    - kullanıcı sorusunu alır ve ilgili bilgiyi veri tabanından getirir. sonra gemini ile cevap üretir.
    - Retrieval:
        - kullanıcı sorusunu sorar -> embedding ile vektörleştir
        - faiss db üzerinden en alakalı içerik (chunk) getirilir
    - Augmentation: (içeriği zenginleştirme)
        - kullanıcı sorusu + prompt + getirilen bilgi (zenginleştirme)
    - Generation: dil modeli bilgiler ile mantıklı yanıt üretir

Plan/Program:
    - DB işlemleri: build_vector_db.py
        - sözleşme belgesi hazırlama
        - bu belgeyi okuma, metin çıkarma, parçalama (chunk), embedding ile sayısal vektörleştirme ve faiss db oluşturma
    - Soru Cevap sistemi: main.py
        - kullanıcı sorusunu sorar, embedding ile sayısal vektörleştirilir, RAG yapılır.

Kütüphaneler:
    - pip install google-generativeai
    - pip install python-dotenv
    - pip install sentence-transformers -> embedding ile sayısal vektörleştirme için gerekli.
    - pip install faiss-cpu -> cpu'da çalışan vektörel database
    - pip install numpy
    - pip install PyMuPDF -> PDF dosyasını yüklemek için gerekli

pip freeze > requirements.txt


Q1: What is the project topic and which AI model will be used?
A: The project is an "LLM-Based Chatbot Project". The primary AI backend will be the Google Gemini model.

Q2: What is the total timeline and how can it be extended?
A: The project duration is strictly 6 months. Any extension requires a mutual written amendment.

Q3: What is the total cost and how is the payment structured?
A: The total cost is a fixed net of 5,000 USD, split into three milestones: 30% advance, 40% upon beta prototype, and 30% upon final acceptance.

Q4: When are Intellectual Property (IP) rights transferred to the Client?
A: All source codes and IP rights are completely and irrevocably transferred to the Client upon receipt of the final payment.
"""


import os
from dotenv import load_dotenv
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer # "all-MiniLM-L6-v2"
import google.generativeai as genai # google'ın gemini modellerini daha pratik kullanmayı sağlar (deprecated)


load_dotenv(r"C:\Projects\google_codes\sozlesme_inceleme_asistani\.env")
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash") # gemini modelini başlat (initialize)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # embedding modeli

# faiss vektörel db index dosyasını yükle (build_vector_db.py dosyasında önceden oluşturup kaydettiğim vektörel database)
index = faiss.read_index("C:\Projects\google_codes\sozlesme_inceleme_asistani\data\sozlesme_ornegi.faiss")

# parçalara ayrılmış (chunklanmış) metin verisini yükle
with open("data/sozlesme_ornegi.pkl", "rb") as f:
    chunks = pickle.load(f)

# kullanıcıdan gelen soruları alır
while True:
    user_message = input(f"kullanıcı:")
    if user_message.lower() in ["quit", "exit", "q"]:
        print(f"çıkış yapılıyor.")
        break
    user_message_embedding = embedding_model.encode([user_message]) # faiss vektörel db'de arama yaparken 2D format istenilir..!

    # faiss vektörel db'de embedding olan user_message'a en yakın 3 chunk aranır ve getirilir
    k = 3 # en yakın 3 chunk
    distances, indices = index.search(np.array(user_message_embedding), k=k) # distances -> bulunan chunk'ların kullanıcı sorusuna ne kadar uzak olduğu (L2 mesafesi)
    # indices -> bulunan chunkların index numaraları

    # bulunan chunkları birleştir ve bağlam (context) oluştur
    retrieved_chunks = [chunks[i] for i in indices[0]] # ilk satırdaki chunk'ları alır ve retrieved_chunks değişkeninde birleştirir.
    context = "\n-----\n".join(retrieved_chunks) # retrieved_chunks listesindeki chunk halindeki metinleri arasına ayırıcı koyarak birleştiriyor.

    # llm'e gönderilecek sistem prompt'u
    prompt = f"""
        You are a contract lawyer AI assistant. Based on the contract context below, answer the user's question clearly.
        Context: {context}
        User message: {user_message}
        Answer:
        """
    response = gemini_model.generate_content(prompt) # gemini 2.5 flash modelinden yanıt alma
    print(f"AI: {response.text.strip()}")