

"""
Problem Tanımı:
    - akıllı müşteri destek sistemi yap
    - sık sorulan sorulara yanıt ver ve belgeye dayalı yanıt sistemi kur
    - müşteriler genelde benzer sorular sorarlar (şifremi unuttum, faturamı nereden alabilirim, iade süresi vb.)
    - Çözüm:
        - .pdf, db, text, json vb. veri formatından sıkça sorulan soruları vektörel database'e dönüştür.
        - kullanıcıdan gelen sorular database'de sorgulanır ve gemma4 (açık kaynak) dil modeli ile türkçe cevaplar üret.

Kullanılan Teknolojiler:
    - langchain: rag mimarisi kurmak için
    - faiss: embedding'leri saklamak için vektörel database
    - ollama: ollama üzerinden gemma4:e4b parametreli modeli kullan
    - streamlit: web arayüzü (UI) oluşturmak

Veri seti: gemini ile veri oluştur.
    - soru: yurt dışı satışlarınız bulunuyor mu
    - cevap: henüz bulunmuyor

Plan/Program:
    - sss içeren pdf dosyası oluştur.
    - kullanıcı bu dosyayı arayüzden yükler.
    - pdf metni chunk'lara ayrılır ve chunk metin verilerine embedding yapılır.
    - kullanıcı soru sorduğunda faiss vektör db'den benzer içerikteki chunk'lar getirilir ve gemma4:e4b dil modeli ile cevap oluşturulur.
    - memory (hafıza) ile konuşma geçmişi saklanır ve sonraki yanıtlara bağlam oluşturulur.

Kütüphaneler:
    - pip install langchain
    - pip install langchain-community
    - pip install langchain-classic
    - pip install langchain-ollama
    - pip install sentence-transformers
    - pip install streamlit
    - pip install faiss-cpu
    - pip install pypdf
    - pip install ollama
"""