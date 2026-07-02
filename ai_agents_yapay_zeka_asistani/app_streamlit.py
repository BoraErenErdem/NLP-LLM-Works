

"""
streamlit ile ui oluştur.
streamlit run app_streamlit.py
"""

import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/ask" # local host ve endpoint
USER_ID = "bee" # demo user

st.set_page_config(page_title="Multi-Tool AI Agent Chat", page_icon="🐜", layout="centered")
st.title("🐜 Llama + Gemini Multi-Tool AI Agent")
st.caption("Gemini 2.5 flash + Llama 3.3:70b versatile + RAG + Math + Discount + Web Search + Currency Converter + Memory")

if "message" not in st.session_state:
    st.session_state["message"] = []

# mesaj gönderme fonksiyonu
def send_message(message: str):
    payload = {
        "user_id": USER_ID,
        "message": message
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response received from the server.")
        else:
            return f"{response.status_code}, {response.text}"
    except Exception as e:
        return f"Bağlantı hatası -> {e}"


user_message = st.chat_input(f"mesaj yazınız...")
if user_message:
    st.session_state["message"].append({"role": "user", "content": user_message})

    with st.spinner("Agent thinking..."):
        response = send_message(user_message)

    st.session_state["message"].append({"role": "assistant", "content": response})

for msg in st.session_state["message"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**Siz:** {msg["content"]}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Agent** {msg["content"]}")