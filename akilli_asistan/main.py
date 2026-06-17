

"""
Problem tanımı:
    - Gemini ile Akıllı Asistan Projesi: notlar ve etkinlikler için akıllı asistan kullan.
    - Google Gemini API kullan.
    - Kural tabanlı notlar ve etkinlikler oluştur.
    - Doğal dilde notlar ve etkinlikler ile konuşabilme. (chatbot)
    - Kısaca asistan, notlara ve etkinkiklere erişim sağlayarak özetleme, bilgi çıkarma ve takvim oluşturma vb. görevleri yapar.

Örnek senaryo:
    - notlar: akşam eve giderken markete uğramayı unutma, kargoyu almayı unutma vb.
    - etkinlik: konser var, haftaya toplantı var vb.
    - chatbot: haftaya ne vardı?, akşam eve giderken yapmam gereken bir şey var mıydı? vb.

Model tanıtımı:
    - google-gemini-2.5-flash modeli

Plan/Program:
    - assistant.py: gemini ile chatbot oluşturma
    - database.py: notlar ve etkinlikleri sqlite içinde depola
    - main.py: bileşenleri bir araya getir

Kütüphaneler:
    - pip install requests
    - pip install python-dotenv

"""