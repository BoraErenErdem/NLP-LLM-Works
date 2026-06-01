

"""
Problem tanımı:
    - Akıllı Doktor Asistanı: kullanıcının sağlıkla ilgili sorularını anlayan ve yanıtlayan LLM tabanlı doktor asistanı (chatbot)
    - LLM: Google Gemini API
    - Kişiselleştirme: Kullanıcının "adını", "yaşını" ve "geçmişini" bilerek ona göre cevap üretmeli.
    - Hafıza (Memory) : Mesaj geçmişini hatırlayarak konuşmayı ona göre sürdürmeli.

Çalışma ortamı:
    - İlk olarak terminal üzerinden çalıştır. (doktor_asistani_terminal.py)
        - test edebilmek için terminal üzerinden sorgu oluştur.
    - FastAPI ile web servisi oluştur. (doktor_asistani_api.py)
        - client.py dosyası ile test senaryosu belirle.
        - swagger üzerinden test et.

Veri seti: RAG yapmayacağım için onun yerine prompt engineering yapıcam.

Model tanıtımı: Google Gemini 2.5 flash
    - API ile iletişim kur ve gerçek zamanlı sağlık önerileri al.


Kütüphaneler:
    - langchain -> llm kütüphanesi, prompt yönetimi, memory, chain yapısı
    - fastapi -> web api geliştirmek
    - uvicorn -> fastapi geliştirmek için sunucu

Kurulumlar:
    - pip install langchain-google-genai
    - pip install python-dotenv
    - pip install langchain
    - pip install fastapi
    - pip install uvicorn

requirements.txt -> kurulum sonrasında pip freeze > requirements.txt oluştur

API KEY -> .env içerisinde
"""


import os # .env içerisinden api key almak
from dotenv import load_dotenv # .env içerisinden GOOGLE_API_KEY almak
from langchain_google_genai import ChatGoogleGenerativeAI # gemini modelleri için langchain ara birimi
from langchain_classic.memory import ConversationBufferMemory # sohbet geçmişini saklamak için hafıza
from langchain_classic.chains import ConversationChain # hafıza + llm zinciri
import warnings
warnings.filterwarnings("ignore")

# .env'den ortam değişkenlerinin (API) yüklenmesi
load_dotenv(r'C:\Projects\google_codes\doktor_asistani\.env')
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError('Google API key is missing.')

# llm modelinin tanımlanması
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7, google_api_key=api_key)
# model= kullanılacak llm modeli # temperature= cevap çeşitliliği (0 -> garantici, 1 -> yaratıcı)

# hafıza (memory) tanımlama
memory = ConversationBufferMemory(return_messages=True) # return_messages=True -> terminale yazdırmasını sağlar.

# llm + memory = chain (zincir) # model artık önceki mesajları hatırlayarak cevap üretiyor
conversation = ConversationChain(llm=llm, memory=memory, verbose=True) # verbose=True -> terminalde işlem akışı detayları görebilirim.

# kullanıcı bilgilerini tanımla (kişiselleştirme)
name = input("Adınız: ")
age = int(input("Yaşınız: "))
history = input("Hasta geçmişi: ") # kullanıcının geçmişi

# prompt tanımlama
intro = (f"Sen bir doktor asistanısın. Hasta {name}, {age} yaşında ve {history} geçmişine sahip." "Sağlık sorunları hakkında konuşmak istiyor." "Yaşına uygun, dikkatli ve nazik tavsiyeler ver; ismiyle hitap et, maksimum 3-5 cümle ile cevap ver.")

# başlangıç mesajını hafızaya kaydet
memory.chat_memory.add_user_message(intro) # sistem prompt'u hafızaya eklenir. model her cevabında bunu bağlam olarak kullanır.
print(f'Merhaba ben bir doktor asistanıyım, size nasıl yardımcı olabilirim?')

# chatbot diyalog döngüsü
while True:
    # kullanıcı mesajını al
    user_message = input(f"{name}: ")
    if user_message.lower() in ["quit", "q"]:
        print(f"Size yardımcı olabildiysem ne mutlu, görüşmek üzere.👋")
        break

    # llm (chatbot) cevabı
    # modelden yanıt al yani gemini'den yanıt al
    reply = conversation.predict(input=user_message)

    # doktor asistanı cevabını yazdır
    print(f"Doktor asistanı: {reply}")