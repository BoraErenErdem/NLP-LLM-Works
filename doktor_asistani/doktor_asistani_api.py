

"""
FastAPI ile Gemini 2.5 flash doktor asistanını web servisine çevir.
Her kullanıcı için ayrı bir memory (sohbet geçmişi) tut.

uvicorn doktor_asistani_api:app --reload
"""


import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # istek ve yanıt şemaları
from dotenv import load_dotenv # .env içerisinden google api key'i okur
from langchain_google_genai import ChatGoogleGenerativeAI # gemini modelleri için langchain ara birimi
from langchain_classic.memory import ConversationSummaryBufferMemory # konuşma uzadıkça eski mesajlar özetlenerek saklanır..!
from langchain_classic.chains import ConversationChain # llm + memory = chain

load_dotenv(r'C:\Projects\google_codes\doktor_asistani\.env')
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing.")

app = FastAPI(title="Gemini 2.5 Doktor Asistanı API")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=api_key)

# her kullanıcı için memory yapılandırması
user_memories: Dict[str, ConversationSummaryBufferMemory] = {} # her kullanıcı için ayrı hafıza tutan boş dict. # tek memory kullanılırsa konuşmalar karışabilir bu yüzden her kullanıcı için ayrı memory oluşturdum..! # str (key) -> kullanıcı adı vb. # ConversationSummaryBufferMemory (value) -> o kullanıcının hafızası

# istek (request) ve yanıt (response) şemaları
class ChatRequest(BaseModel): # kullanıcının gönderdiği mesaj
    name: str
    age: int
    message: str

class ChatResponse(BaseModel): # modelin döndürdüğü yanıt
    response: str

# sohbet endpointi oluşturma
@app.post("/chat", response_model=ChatResponse)
async def chat_with_doctor(request: ChatRequest):
    try:
        # kullanıcıya özel memory oluşturma
        if request.name not in user_memories: # eğer memory'de bu kullanıcı yoksa yeni memory oluşturur
            user_memories[request.name] = ConversationSummaryBufferMemory(llm=llm, return_messages=True)
        memory = user_memories[request.name] # memory'de bu kullanıcı zaten varsa kullanıcıyı getirir

        # ilk konuşma (eğer memory boşsa giriş bağlamı ekle)
        if len(memory.chat_memory.messages) == 0: # memory.chat_memory.messages -> memory'deki tüm mesaj geçmişi listesidir. # eğer memory boşsa ilk kez girdiği için prompt eklenir..!
            intro = (f"Sen bir doktor asistanısın. Hasta: {request.name}, {request.age} yaşında. Sağlık sorunları hakkında konuşmak istiyor. Yaşına uygun, dikkatli ve nazik tavsiyeler ver. Kullanıcıya ismiyle hitap et.")
            memory.chat_memory.add_user_message(intro)

        # llm + memory = chain (zincir) yani sohbet zinciri oluştur
        conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

        # modelden yanıt (response) al
        reply = conversation.predict(input=request.message)

        # terminale memory'i yazdır
        print(f"Memory for {request.name}")
        for idx, m in enumerate(memory.chat_memory.messages, start=1):
            print(f"{idx:02d}. {m.type.upper()}: {m.content}")
        print('--------------------------------------------------')

        # api yanıtını (response) return et
        return ChatResponse(response=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
