

"""
Problem Tanımı:
    - akıllı müşteri destek sistemi yap
    - sık sorulan sorulara yanıt ver ve belgeye dayalı yanıt sistemi kur
    - müşteriler genelde benzer sorular sorarlar (şifremi unuttum, faturamı nereden alabilirim, iade süresi vb.)
    - Çözüm:
        - .pdf, db, text, json vb. veri formatından sıkça sorulan soruları vektörel database'e dönüştür.
        - kullanıcıdan gelen sorular database'de sorgulanır ve gemma4 (açık kaynak) dil modeli ile türkçe cevaplar üret.

Kullanılan Teknolojiler:
    - langchain: rag mimarisi kurmak için
    - faiss: embedding'leri saklamak için vektörel database
    - ollama: ollama üzerinden gemma4:e4b parametreli modeli kullan
    - streamlit: web arayüzü (UI) oluşturmak

Veri seti: gemini ile veri oluştur.
    - soru: yurt dışı satışlarınız bulunuyor mu
    - cevap: henüz bulunmuyor

Plan/Program:
    - sss içeren pdf dosyası oluştur.
    - kullanıcı bu dosyayı arayüzden yükler.
    - pdf metni chunk'lara ayrılır ve chunk metin verilerine embedding yapılır.
    - kullanıcı soru sorduğunda faiss vektör db'den benzer içerikteki chunk'lar getirilir ve gemma4:e4b dil modeli ile cevap oluşturulur.
    - memory (hafıza) ile konuşma geçmişi saklanır ve sonraki yanıtlara bağlam oluşturulur.

Kütüphaneler:
    - pip install langchain
    - pip install langchain-community
    - pip install langchain-classic
    - pip install langchain-ollama
    - pip install sentence-transformers
    - pip install streamlit
    - pip install faiss-cpu
    - pip install pypdf
    - pip install ollama
"""



from langchain_ollama.chat_models import ChatOllama # ollama + gemma4:e4b
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.vectorstores import FAISS # vector_db
from langchain_classic.memory import ConversationSummaryBufferMemory # memory
from langchain_classic.embeddings import HuggingFaceEmbeddings # LaBSE embedding modeli


llm = ChatOllama(model="gemma4:e4b", temperature=0.2) # ollama üzerinden kullanılacak on-prem llm modeli

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE") # LaBSE embedding modeli

vector_db = FAISS.load_local("faq_vector_store", embeddings, allow_dangerous_deserialization=True) # önceden oluşturulmuş vector_db'yi yükle
# allow_dangerous_deserialization=True -> .pkl formatında dosya yüklerken hata oluşmasını önler.

memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True) # memory_key="chat_history" -> memory'nin hangi isimle saklanacağını belirtir. ConversationalRetrievalChain fonksiyonu memory'i ararken "chat_history" anahtar kelimesini kullanır..! RAG + memory yapılırken memory_key="chat_history" olarak belirtilir..!

qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k":3}), memory=memory, verbose=True)
# ConversationalRetrievalChain.from_llm() -> RAG + memory + llm'i tek zincirde birleştirir.
# retriever=vector_db.as_retriever(search_kwargs={"k":3}) -> vector_db'den getirici nesnesi getirir. as_retriever() ile vector_db'de arama yapar. search_kwargs={"k":3} ifadesi her arama yapıldığında en yakın 3 chunk'ı getirir..!

# test
print("müşteri destek botuna hoşgeldiniz.")
while True:
    user_input = input("kullanıcı: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print(f"byebyeee.")
        break
    response = qa_chain.invoke({"question": user_input})
    print(f"müşteri destek botu: {response["answer"]}")