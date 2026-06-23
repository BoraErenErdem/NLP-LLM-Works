

"""
- DB işlemleri: build_vector_db.py
    - sözleşme belgesi hazırlama
    - bu belgeyi okuma, metin çıkarma, parçalama (chunk), embedding ile sayısal vektörleştirme ve faiss db oluşturma
"""


import fitz # PyMuPDF kütüphanesi
from sentence_transformers import SentenceTransformer # embedding için # hugging face'ten sentence-transformers/all-MiniLM-L6-v2 modelini kullan.
import faiss # vector database
import numpy as np
import pickle # vector db'i kaydetmek için

pdf_file_path = r'C:\Projects\google_codes\sozlesme_inceleme_asistani\data\software_development_agreement.pdf'

# .pdf'den metin dönüşümü yapacak fonksiyon
def extract_text_from_pdf(pdf_path): # pdf dosyasından metin çıkarımı
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text = text + page.get_text()
    return text

# print(f'{extract_text_from_pdf(r'C:\Projects\google_codes\sozlesme_inceleme_asistani\data\software_development_agreement.pdf')}')

# uzun metni daha küçük parçalara (chunklara) böl
def chunk_text(text, max_length=500): # metni belirtilen karakter uzunluğuna böl
    chunks = [] # bölünmüş parçaların listesi
    current = "" # şu anda biriktirilen parça
    for line in text.split("\n"): # metni satır satır gezer
        if len(current) + len(line) > max_length: # mevcut parça dolduysa kaydet ve yeni parçaya başla  # line -> o satırdaki karakterler # len(line) -> o satırdaki karakter sayısı
            chunks.append(current.strip())
            current = line.strip()
        else: # mevcut parçaya satırı ekle
            current += " " + line.strip()
    if current:
        chunks.append(current.strip())
    print(len(chunks))
    return chunks

# text = extract_text_from_pdf(r'C:\Projects\google_codes\sozlesme_inceleme_asistani\data\software_development_agreement.pdf')
# print(f'{chunk_text(text)}')

text = extract_text_from_pdf(pdf_file_path) # pdf'den metin çıkarımı
chunks = chunk_text(text) # metni chunklara bölme
model = SentenceTransformer("all-MiniLM-L6-v2") # hugging face'ten kullanılan embedding modeli
embeddings = model.encode(chunks) # her chunk için embedding yani sayısal vektör temsili oluştur
print(f'{embeddings.shape}') # (10, 384)

dimension = embeddings.shape[1] # embedding vektör boyutu (384)
index = faiss.IndexFlatL2(dimension) # L2 norm (euclidean distance) kullanarak benzerlik arar ve bu boyutta çalışan boş vektör database'i oluşturur.
index.add(np.array(embeddings)) # embedding'leri index'e kaydeder ve tüm embedding'ler database'e kaydedilir.

# faiss indexi ve chunkları kaydet
faiss.write_index(index, "data/sozlesme_ornegi.faiss")
with open("data/sozlesme_ornegi.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f'faiss index ve chunklar kaydedildi.')