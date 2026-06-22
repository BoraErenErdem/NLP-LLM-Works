

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


from assistant import get_gemini_response, detect_intent # assistant.py dosyasından gemini api yanıtını alan get_gemini_response fonksiyonu ve detect_intent fonksiyonu ile niyet anlama
from database import initialize_db, add_notes, add_events, get_notes, get_events, add_special_days, get_special_days # database.py dosyasından veri tabanı için gerekli olan fonksiyonlar

initialize_db() # veri tabanını başlatır
print(f"Akıllı Asistan'a hoşgeldiniz.\n"
        f"Komutlar -> not ekle | etkinlik ekle| özel gün ekle | notlarımı göster | etkinliklerimi göster | özel günlerimi göster | sohbet et | çıkış")

# kullanıcıdan sürekli komut almak için sonsuz döngü
while True:
    komut = input("Komut: ").strip().lower() # kullanıcıdan komut alır ve alınan komutta boşluklar veya büyük harf varsa boşlukları kırpar ve hepsini küçük harfe çevirir.
    if komut == "not ekle":
        content = input("not içeriği:") # kullanıcıdan not içeriği al
        add_notes(content)
        print(f'notunuz başarıyla kaydedildi.')

    elif komut == "etkinlik ekle":
        event = input("etkinlik içeriği:")
        event_date = input("etkinlik tarihi:")
        add_events(event, event_date)
        print(f'etkinliğiniz başarıyla kaydedildi.')

    elif komut == "özel gün ekle":
        explanation = input(f"özel gün içeriği:")
        explanation_date = input(f"özel gün tarihi:")
        add_special_days(explanation, explanation_date)
        print(f"özel gün başarıyla eklendi.")

    elif komut == "notlarımı göster":
        notes = get_notes() # veri tabanından notları getirir
        if notes:
            print(f'Notlarım:')
            for content, created_at in notes:
                print(f"\t- [{created_at}] {content}")
        else:
            print(f'Notunuz bulunmamaktadır.')

    elif komut == "etkinliklerimi göster":
        events = get_events()
        if events:
            print(f"Etkinliklerim:")
            for event, event_date in events:
                print(f'\t- [{event_date}] {event}')
        else:
            print(f"Etkinlikleriniz bulunmamaktadır.")

    elif komut == "özel günlerimi göster":
        special_days = get_special_days()
        if special_days:
            print(f"Özel günlerim:")
            for explanation, explanation_date in special_days:
                print(f"\t- [{explanation_date}] {explanation}")
        else:
            print(f"Özel günleriniz bulunmamaktadır.")

    elif komut == "sohbet et":
        user_message = input(f"Kullanıcı: ").strip() # kullanıcıdan serbest metin alma
        intent = detect_intent(user_message) # kullanıcı niyeti yani not özeti, etkinlik özeti, günlük konuşma vb.

        if intent == "not_ozet":
            notes = get_notes()
            if not notes:
                print(f"Özetlenecek not bulunamadı.")
                continue
            all_notes_text = "\n".join([f"- {note[0]}" for note in notes]) # tüm notları birleştirir ve text haline getirir
            prompt = f"Aşağıda bulunan notlar doğrultusunda kullanıcı sorusunu yanıtlar mısın? Eğer notlarda kullanıcı sorusuna cevap yoksa bilmediğini kibarca belirt. notlar: {all_notes_text}, kullanıcı sorusu: {user_message}"
            response = get_gemini_response(prompt) # gemini'den cevap iste
            print(f"Notlar hakkında: {response}")

        elif intent == "etkinlik_ozet":
            events = get_events()
            if not events:
                print(f"Özetlenecek etkinlik bulunamadı.")
                continue
            all_events_text = "\n".join([f"- {event[1]}: {event[0]}" for event in events]) # tüm etkinlikleri birleştirir ve text haline getirir
            prompt = f"Aşağıda takvime göre kullanıcı sorusunu yanıtlar mısın? Eğer etkinlikleride kullanıcı sorusuna cevap yoksa bilmediğini kibarca belirt. takvim: {all_events_text}, kullanıcı sorusu: {user_message}"
            response = get_gemini_response(prompt)
            print(f'Etkinlikler hakkında: {response}')

        elif intent == "ozelgun_ozet":
            special_days = get_special_days()
            if not special_days:
                print(f"Özetlenecek özel gün bulunamadı.")
            all_special_days_text = "\n".join([f"{s_day[1]}, {s_day[0]}" for s_day in special_days])
            prompt = f"Kayıtlı bulunan özel günlere göre kullanıcı sorularını yanıtla. Eğer özel günlerde kullanıcı sorusuna cevap yoksa bilmediğini kibarca söyle. özel günler: {all_special_days_text}, kullanıcı sorusu: {user_message}"
            response = get_gemini_response(prompt)
            print(f"Özel günler hakkında: {response}")

        else: # normal (diğer her şey)
            reply = get_gemini_response(user_message)
            print(f"Akıllı Asistan: {reply}")

    elif komut == "çıkış":
        print(f"İyi günler dilerim..👋")
        break
    else:
        print(f"Hatalı komut..!")