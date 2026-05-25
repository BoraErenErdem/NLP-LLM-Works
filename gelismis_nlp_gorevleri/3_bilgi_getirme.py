

"""
Amaç:
    - BERT modelini kullanarak metin benzerliği (semantic similarity) analizi yap.
    - kullanıcı sorgu (query) cümlesi ile bir dizi belgenin anlamca ne kadar benzer olduğunu ölç.
    - her metin BERT modelinden elde edilen embedding (vektör temsili) ile temsil edilir.
    - cosine similarity ile benzerlik ölçümü gerçekleştir.

Adımlar:
    - kütüphaneleri yükle
    - BERT modelini ve tokenizer'ı yükle
    - örnek belgeleri ve sorgu (query) cümlesini oluştur
    - her metni embedding vector haline getir
    - sorgu ve belgeler arasındaki benzerliği cosine similarity ile hesapla
    - en benzer belgeyi belirle ve yazdır

pip install transformers
pip install torch
pip install scikit-learn
pip install numpy
"""


import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity # cosine_similarity -> benzerlik hesaplamak için kullanılır


model_name = 'bert-base-uncased' # küçük boyutlu ingilizce metinlerde kullanılan BERT modeli
model = BertModel.from_pretrained(model_name) # önceden eğitilmiş model yüklenir
tokenizer = BertTokenizer.from_pretrained(model_name) # BERT modeli ile uyumlu tokenizer

documents = [ # veri
    "Machine Learning is a field of artifical intelligence",
    "Natural Language Processing involves understanding human language",
    "Artifical intelligence encompasses machine learning and natural language processing (NLP)",
    "Deep Learning is a subset of machine learning",
    "Data science combines statistic, data analysis and machine learning",
    "I like shopping"
]

query = "What is deep learning?" # kullanıcı sorusu

def get_embedding(text): # verilen metni BERT modeli kullanarak sayısal vektörlere dönüştür. 1.tokenization 2.modeli çalıştır 3.embedding yap 4.çıktıyı np formatına çevir
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True) # padding -> giriş uzunluğunu batch içindeki en uzun sequence'e eşitler..!
    outputs = model(**inputs) # ** -> dictionary içindeki verileri fonksiyon parametrelerine açar (yani unpack eder..!)
    # outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']) # bu ifade de aynı şeydir..!
    last_hidden_state = outputs.last_hidden_state # her token'ın embedding'ini (sayısal vektörü)
    embedding = last_hidden_state.mean(dim=1) # dim=1 -> token ekseninde işlem yapar. tüm token embedding'lerin (512) ortalamasını alır ve tek cümle vektörü oluşturur. (mean pooling)
    return embedding.detach().numpy() # detach() -> tensör'ü hesaplama grafiğinden ayırır ve gradient takibini kapatır.

doc_embedding = np.vstack([get_embedding(doc) for doc in documents]) # birden fazla cümle olduğundan her belge için embedding yapılır. # np.vstack() -> array'leri alt alta birleştirir.
query_embedding = get_embedding(query) # tek cümle olduğu için vstack() kullanmak yerine direkt get_embedding() fonksiyonunu kullandım.
# eğer query kısmında birden fazla cümle olsaydı o zaman doc_embedding gibi np.vstack() kullanıp yapmam gerekirdi..!

similarities = cosine_similarity(query_embedding, doc_embedding) # cosine_similarity için sonuç 1 ise tamamen benzer, 0 ise tamamen alakasız

for i, score in enumerate(similarities[0]):
    print(f'Documents: {documents[i]} \n Similarity Score: {score:.4f}\n')

most_similar_index = similarities.argmax()
print(f'most similar document: {documents[most_similar_index]}')
