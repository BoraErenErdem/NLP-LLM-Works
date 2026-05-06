

"""

imdb film yorumları üzerinden Word2Vec ve FastText tabanlı kelime vectorleri üretmek ve k-means algoritmasını kullanarak kümelere ayırma
Adım:
    - veri setini .csv olarak yükle
    - metinleri temizle (text cleaning) (küçük büyük harf dönüşümü, noktalama ve özel karakterleri temizleme, kısa kelimelerin kaldırılması)
    - stopwordsleri kaldır
    - tokenizasyon
    - Word2Vec ve FastText modeli tanımla ve word embedding yap
    - ilk 500 kelime k-means ile 2'li kümeleme yap
    - PCA kullanarak vektörleri 50 boyuttan 2 boyuta indirge
    - sonuçları 2D'de görselleştir

pip install gensim

"""


import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# region word2vec modeli için word embeddings
df = pd.read_csv(r'C:\Projects\google_codes\metin_temsili\IMDB Dataset.csv')
print(df.head())
df.columns

review = df['review']

def text_cleaning(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = " ".join([i for i in text.split() if len(i) > 2])
    stopwords_eng = set(stopwords.words('english'))
    text = " ".join([i for i in text.split() if i not in stopwords_eng])
    return text

clean_review = [text_cleaning(i) for i in review] # review'ları temizleme
print(clean_review[:5])

tokenize_review = [simple_preprocess(i) for i in clean_review] # temizlenmiş review'lara tokenizasyon
print(tokenize_review[:5])

word2vec_model = Word2Vec(sentences=tokenize_review, vector_size=50, window=5, min_count=1, sg=0, workers=10) # workers= -> kaç cpu çekirdeği kullanılacağını belirler (varsayılan = 3)
fasttext_model = FastText(sentences=tokenize_review, vector_size=50, window=5, min_count=1, sg=0, workers=10)

word_vectors = word2vec_model.wv # kelime vektörleri
vocabulary = list(word_vectors.index_to_key)[:500] # vocabulary (içinden ilk 500 kelime seçildi)
vectors = [word_vectors[i] for i in vocabulary] # ilk 500 kelimenin vector boyutları 50 yapıldı

kmeans = KMeans(n_clusters=2, random_state=42) # 2 küme kullanılacak
kmeans.fit(vectors)
cluster_labels = kmeans.labels_ # her kelime için küme etiketi

pca = PCA(n_components=2, random_state=42) # 2D'ye indirgenicek
reduced_vectors = pca.fit_transform(vectors)

plt.figure(figsize=(12, 7))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=cluster_labels, cmap='viridis')

cluster_centers = pca.transform(kmeans.cluster_centers_) # küme merkezlerinin işaretlenmesi (burada 50 boyutlu kmeans cluster centersları pca ile 2D'ye indirgeyip merkezlerini belirle)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=150, label='Center')

for i, word in enumerate(vocabulary):
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word, fontsize=8)

plt.legend(loc='upper left')
plt.title('Word2Vec + KMeans Clustering')
plt.show()
# endregion




# region word2vec ve fasttext için custom function ve kmeans + pca ile word embeddings
df = pd.read_csv(r'C:\Projects\google_codes\metin_temsili\IMDB Dataset.csv')
print(df.head())

review = df['review']
print(review[:5])

def text_cleaner(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = " ".join([i for i in text.split() if len(i) > 2])
    stopwords_eng = set(stopwords.words('english'))
    text = " ".join([i for i in text.split() if i not in stopwords_eng])
    return text

filtered_review = [text_cleaner(i) for i in review]
print(filtered_review[:5])

tokenize_review = [simple_preprocess(i) for i in filtered_review]
print(tokenize_review[:5])

word2vec_model = Word2Vec(sentences=tokenize_review, vector_size=50, window=5, min_count=1, sg=0, workers=10)
fasttext_model = FastText(sentences=tokenize_review, vector_size=50, window=5, min_count=1, sg=0, workers=10)

def word_embedding_plots(model, title):
    word_vectors = model.wv
    vocabulary = list(word_vectors.index_to_key)[:500]
    vectors = [word_vectors[i] for i in vocabulary]
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(vectors)
    cluster_labels = kmeans.labels_
    pca = PCA(n_components=3, random_state=42)
    reduced_vectors = pca.fit_transform(vectors)
    cluster_centers = pca.transform(kmeans.cluster_centers_)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], c=cluster_labels, cmap='viridis')
    for i, word in enumerate(vocabulary):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=10)

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='x', s=150, label='Center')
    plt.legend()
    plt.title(title)
    plt.show()


word_embedding_plots(word2vec_model, 'Word2Vec Model + KMeans with PCA')
word_embedding_plots(fasttext_model, 'FastText Model + KMeans with PCA')
# endregion