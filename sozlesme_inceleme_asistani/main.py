

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
"""



"""
Q1: What is the project topic and which AI model will be used?
A: The project is an "LLM-Based Chatbot Project". The primary AI backend will be the Google Gemini model.

Q2: What is the total timeline and how can it be extended?
A: The project duration is strictly 6 months. Any extension requires a mutual written amendment.

Q3: What is the total cost and how is the payment structured?
A: The total cost is a fixed net of 5,000 USD, split into three milestones: 30% advance, 40% upon beta prototype, and 30% upon final acceptance.

Q4: When are Intellectual Property (IP) rights transferred to the Client?
A: All source codes and IP rights are completely and irrevocably transferred to the Client upon receipt of the final payment.
"""