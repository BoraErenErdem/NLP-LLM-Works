

"""
web üzerinde çalışan chatbot ekranı geliştir.
streamlit framework'ü kullan.
    - pip install streamlit

streamlit run akilli_turizm_rehberi_streamlit_streaming.py -> terminalde streamlit çalıştırmak (local host olarak 8501 portunda çalıştırır)
"""


import streamlit as st # streamlit ile web arayüzü oluşturmak
from langchain_ollama.chat_models import ChatOllama
from langchain_classic.schema import SystemMessage, HumanMessage
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.callbacks.base import BaseCallbackHandler # streaming'de streamlit ile çalışmak için
from typing import Any

# streamlit için özel streaming callback tanımı
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder # streamlit içindeki mesaj kutusu
        self.final_text = "" # biriken metin

    def on_llm_new_token(self, token:str, **kwargs:Any) -> None:
        self.final_text += token # yeni token'ı metne ekle
        self.placeholder.markdown(self.final_text + " ") # canlı olarak ekranda güncelle

# başlık ve açıklamalar
st.set_page_config(page_title="Akıllı Turizm Rehberi", page_icon="🗺️")
st.title("🗺️ Akıllı Turizm Rehberi")
st.markdown("Türkiye'nin dört bir yanındaki turistik yerler hakkında bilgi almak için sorular sorabilirsiniz.")

llm = ChatOllama(model="gemma3:4b")

# session state -> streamlit için değişkenleri oturum boyunca tutar..!
if 'memory' not in st.session_state:
    st.session_state.llm = llm # llm modelini de session'a ekledim. bu sayede model bir kez oluşturulduktan sonra session'da tutuluyor. (eklemeseydim her etkileşimde modeli baştan yükleyecekti ve bu da yavaşlığa sebep olacaktı..!)
    st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True)

# mesaj kutusu -> kullanıcıdan gelen mesaj
user_input = st.chat_input("Bir şehir, mekan ya da yemek aktivitesi sorabilirsiniz.")

# sohbet geçmişini arayüzde gösterir. yani tüm mesajları sırayla gezip ekrana yazdır
for msg in st.session_state.memory.chat_memory.messages: # session_state'deki memory'deki tüm mesajlar için.
    if isinstance(msg, HumanMessage): # isinstance() -> mesajın türünü kontrol eder
        with st.chat_message("Kullanıcı"): # kimin yazdığını belirtir (kullanıcı veya ai)
            st.markdown(msg.content) # mesaj içeriği
    else:
        with st.chat_message("Akıllı Turizm Rehberi"):
            st.markdown(msg.content)

if user_input:
    st.session_state.memory.chat_memory.add_user_message(user_input) # yeni gelen kullanıcı mesajını session_state'deki memory'e kaydet
    with st.chat_message("Kullanıcı"):
        st.markdown(user_input)

    with st.chat_message("Akıllı Rehber"):
        response_placeholder = st.empty() # streamlit'deki geçici mesaj kutusu
        stream_handler = StreamHandler(response_placeholder)

        llm = ChatOllama(model="gemma3:4b", streaming=True, callbacks=[stream_handler]) # streaming=True -> modeli token token üreten moda geçirir
        # callbacks=[stream_handler] -> her token geldiğinde StreamHandler'ı çağırır

        # tüm konuşmayı modele verecek şekilde liste içinde oluştur -> [system message] + [memory] + [human message]
        messages = [
            SystemMessage(content="Sen bir turizm rehberisin. Kullanıcılara Türkiye'deki şehirler, tarihi yerler, yöresel yemekler, ulaşım ve tatil önerileri yap.")
        ] + st.session_state.memory.load_memory_variables({})['history'] + [HumanMessage(content=user_input)]

        response = llm.invoke(messages) # llm modelinin yanıtı

        st.session_state.memory.chat_memory.add_ai_message(response.content) # llm modelinin ürettiği cevabı session_state'deki memory'e ekliyorum..!