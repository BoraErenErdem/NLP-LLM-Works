

"""
terminal üzerinden fastapi web sunucusu ile sohbet gerçekleştir. (post request atarak)
api endpointi: /chat
"""


import requests # http istekleri yapmak için kullanılan kütüphane

API_URL = "http://127.0.0.1:8000/chat" # fastapi sunucusunun çalıştığı adres ve endpoint

# başlangıçta kullanılan bilgiler
name = input("Adınız: ")
age = int(input("Yaşınız: "))
history = input("Hasta geçmişi: ")
print(f"Sohbet başladı. Çıkmak için quit yazın ya da q tuşuna basın.")

# kullanıcıdan mesajı alıp sunucuya gönderen döngü
while True:
    user_message = input(f"{name}: ")
    if user_message.lower() in ["quit", "q"]:
        print(f"Size yardımcı olabildiysem ne mutlu, görüşmek üzere.👋")
        break

    # API'ye gönderilecek veri paketi
    payload = {
        "name": name,
        "age": age,
        "history": history,
        "message": user_message
    }

    try:
        # fastapi sunucusuna post request at ve timeout belirle (30 sn bekle)
        response = requests.post(API_URL, json=payload, timeout=30) # payload verisini json formatında sunucuya gönderir..!

        if response.status_code == 200: # eğer istek başarılıysa
            print(f"Doktor asistanı: {response.json()["response"]}")
        else:
            print(f"Hata", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print(f'Bağlantı hatası.')