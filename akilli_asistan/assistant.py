

import os # ortam değişkenleri ve dosya yolu
from dotenv import load_dotenv # .env dosyasından api key'i yüklemek
import requests # http istekleri yönetebilmek için

load_dotenv(r'C:\Projects\google_codes\akilli_asistan\.env')
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError('Gemini API key not found!')

# gemini 2.5 flash modelinin api url'i
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" # gemini url adresi

# api çağrısı için gerekli http başlıkları # headers -> kimlik kartı
headers = {
    "Content-Type": "application/json", # json formatında veri gönderileceğini belirtir
    "X-Goog-Api-Key": api_key # yetkilendirme için tanımlanan api_key
}

# gemini api'sine prompt gönderip yanıt alan fonksiyon
def get_gemini_response(prompt:str):

    # api'ye gönderilecek olan json yapısı
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt} # kullanıcıdan gelen mesajı içeren bölüm
                ]
            }
        ]
    }

    # gemini api'ye http post isteği gönder
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200: # istek başarılıysa (http 200)
        try:
            result = response.json() # json formatındaki yanıtı dictionary'e çevir
            return result["candidates"][0]["content"]["parts"][0]["text"] # payload veri paketindeki verileri teker teker açarak text kısmındaki cevaba erişiyorum..!
        except Exception as e:
            return f"Yanıt hatasi {e}"
    else: # istek başarısızsa (hata varsa)
        return f"API hatası: {response.status_code}: {response.text}"

def detect_intense(): # kullanıcıdan gelen mesajı niyet sınıflandırması ile sınıflandırmak
    pass

if __name__ == "__main__":
    user_input = input("Kullanıcı Sorusu: ") # terminal üzerinden girdi almak
    yanit = get_gemini_response(user_input)
    print(f'Akıllı Asistan: {yanit}')