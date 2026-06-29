
"""
streamlit run streamlit_app.py
"""


import tempfile # geçici dosya işlemleri
import streamlit as st
from langchain_ollama.chat_models import ChatOllama
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationSummaryBufferMemory

st.set_page_config(page_title="Müşteri Destek Botu", page_icon="🤖")
st.title("Müşteri Destek Botu (RAG + memory)")
st.write("Bir PDF dosyası yükleyin, içeriğine dair sorular sorun. Türkçe desteklidir.")

uploaded_file = st.file_uploader("PDF dosyanızı yükleyin.", type="pdf", key="pdf_uploader")
if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        with st.spinner("pdf işleniyor..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: # yüklenen pdf'i geçici dosyaya yazdır
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name # geçici dosyanın yolu

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)
            print(f"len docs -> {len(docs)}")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
            vector_db = FAISS.from_documents(docs, embedding_model)
            llm = ChatOllama(model="gemma4:e4b", temperature=0.2)
            memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
            qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k":3}), memory=memory, verbose=True)# RAG + memory + llm chain
            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
            st.session_state.last_uploaded_name = uploaded_file.name
        st.success("PDF başarıyla işlendi.")

if "qa_chain" in st.session_state: # eğer pdf işlendiyse
    user_input = st.text_input("Sorunuzu yazınız") # kullanıcı sorusunu al

    if user_input:
        response = st.session_state.qa_chain.invoke(user_input) # langchain zincirine soruyu gönder
        st.session_state.chat_history.append(("🗣️", user_input)) # kullanıcı mesajını geçmişe ekler
        st.session_state.chat_history.append(("🤖", response["answer"])) # ai yanıtını geçmişe ekler

    if st.session_state.chat_history:
        st.subheader("Sohbet geçmişi")
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}**: {msg}")