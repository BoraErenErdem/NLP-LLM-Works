

"""
fastapi üzerinde çalışan multi-tool için istemci oluştur.
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/ask"

USER_ID = "bee"

def send_message(message: str):
    payload = {
        "user_id": USER_ID,
        "message": message
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # hata varsa yani statsu != 200 ise hata çıkartır.
        data = response.json()
        print(f"Soru: {message}")
        print(f"Cevap: {data.get("response")}")
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    print(f"Multi-Tool Agent started.")
    while True:
        user_message = input("User: ")
        send_message(user_message)