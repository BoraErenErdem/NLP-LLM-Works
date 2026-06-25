

"""
1. sss pdf oluştur ve yükle
2. pdf'ten metin çıkarımı yap
3. metin verilerini chunk et
4. chunk metin verilerine embedding yap
5. vector_db oluştur
6. oluşuturlan vector_db'yi kaydet
"""


from langchain_classic.embeddings import HuggingFaceEmbeddings # langchain uyumlu hugging face'teki embedding sınıfı
from langchain_classic.vectorstores import FAISS # vektör db
from langchain_classic.document_loaders import PyPDFLoader # pdf dosyasından metin çıkarımı
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter # metni anlamlı chunk'lara ayırma işlemi

# pdf dosyasını yükleme ve metin çıkarımı
loader = PyPDFLoader("musteri_destek_faq.pdf") # dosyayı okur
documents = loader.load() # metin çıkarımı yapar
#print(documents)

# metin verilerini chunk'lara ayırma
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # chunk_size= -> her chunk'ın içereceği max karakter sayısı # chunk_overlap= -> her chunk'ın bir önceki chunk ile kesişim karakteri sayısı (yani her chunk bir önceki chunk'tan 50 karakter alabilir)
docs = text_splitter.split_documents(documents) # metin verilerini chunk'lara ayrılır

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE") # sentence-transformers/LaBSE -> hugging face'teki LaBSE modelini kullan. LaBSE -> çok dilli cümle embedding modelidir. 109 farklı dilde cümleleri aynı vektör uzayında temsil edebilir. Farklı dillerdeki anlamca aynı cümleler birbirine yakın vektörler üretir.

# chunk olmuş metin verilerini embedding ile sayısal vektör temsili haline getir ve index oluşturarak faiss vector db'de depola
vector_db = FAISS.from_documents(docs, embeddings) # vektör db için chunk olmuş metin verisini ve embedding modelini verdim..!

# oluşturulan vector_db'yi kaydet
vector_db.save_local("faq_vector_store")

print(f"embedding ve vektör database oluşturuldu.")