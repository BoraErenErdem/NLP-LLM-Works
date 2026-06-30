

"""
RAG: bilgi getirme için musteri_destek_faq.pdf dosyasını kullan. dosyayı oku, metni çıkar, chunk et, embedding yap, faiss vector db oluştur, rag + llm'i tek chain'de birleştir.
"""


from langchain_classic.tools import Tool # Tool -> karmaşık kurulum gerektiren parametrik tool olacağı zaman yapılır. çünkü @tool dekoratörü direkt tool nesnesi olarak başlatır fakat Tool() fonksiyonu önceden hazırlanmış işlemleri (pdf okuma, embedding, chain kurma vb.) tamamladıktan sonra sonucu bir tool nesnesine sarmalar..!
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain


def create_rag_tool(pdf_path: str, llm): # bir kez çalışıcak olan rag fabrikası.
    """
    belirtilen pdf belgesini yükleyip faiss vector db ile arama yapılabilir hale getirir.
    bu örnekte: "musteri_destek_faq.pdf" müşteri destek sss belgesidir.
    """
    print("RAG başlatıldı")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    chunk_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = chunk_text.split_documents(documents)
    print(f"len docs: {len(docs)}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    vector_db = FAISS.from_documents(docs, embedding_model)
    rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k":3}), return_source_documents=False)
    # return_source_documents=False -> yanıt verirken kaynak belgelerinin yani chunk'ların döndürülüp döndürülmemesini belirtir.
    def rag_query(query: str): # agent her soru sorduğunda tekrar tekrar çalışacak olan fonksiyon.
        """
        soruya göre pdf'ten en alakalı bilgiyi getirir.
        """
        response = rag_chain.invoke({"question": query, "chat_history": []})
        return f"belgede bulunan bilgi: {response}"
    return Tool(name="rag_tool", func=rag_query, description="müşteri destek sss belgesinden bilgi araması yapar.") # func=rag_query -> agent her seferinde önceden hazırlanmış rag_chain'i kullanır..!