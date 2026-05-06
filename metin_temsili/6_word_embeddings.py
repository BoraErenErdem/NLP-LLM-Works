

"""

Küçük veri seti üzerinden Word embedding sonra PCA ile görselleştirme
    - Word2Vec -> google
    - FastText -> facebook

Adımlar:
    - cümle veri seti oluştur
    - text preprocessing: cümleleri tokenlara çevir
    - Word2Vec ve FastText modellerini eğit
    - her iki modelden elde edilen vektörleri PCA analizi ile 3D'ye indirge
    - kelime vektörlerini 3D olarak görselleştir

pip install pandas, matplotlib, scikit-learn, gensim

"""


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # PCA ile boyut indirgeme
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

cumleler = [
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsız hareket etmeyi severler.",
    "Köpekler sadık ve dost canlısı hayvanlardır.",
    "Hayvanlar insanlar için iyi arkadaşlardır.",
    "Türkiye'nin başkenti Ankara'dır.",
    "Türkiyede Ankara ve Gaziantep'in yemekleri çok güzel."
]

# simple_preprocess() ile cümlelerin tokenize edilmesi (küçük harf ve noktalama işaretleri temizliği otomatik olarak yapılır..!)
tokenize_cumleler = [simple_preprocess(c) for c in cumleler]
print(f'{tokenize_cumleler}')


# Word2Vec modeli
word2vec_model = Word2Vec(
    sentences=tokenize_cumleler, # eğitim verisi
    vector_size=50, # vektör boyutu
    window=5, # pencere boyutu
    min_count=1, # en az kaç defa geçen kelimelerin alınacağını belirler
    sg=0 # CBOW algoritması
)


# FastText modeli
fasttext_model = FastText(
    sentences=tokenize_cumleler,
    vector_size=50,
    window=5,
    min_count=1,
    sg=0
)


# PCA görselleştirmesi için custom function
def plot_word_embeddings(model, baslik):
    kelime_vector = model.wv # modelin kelime vektörlerinin alınması
    kelimeler = list(kelime_vector.index_to_key)[:1000] # ilk 1000 kelimeyi al
    vectors = [kelime_vector[w] for w in kelimeler]

    pca = PCA(n_components=3) # 3D'ye indirgeme
    indirgenmis_vector = pca.fit_transform(vectors)

    # görselleştirme
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(indirgenmis_vector[:, 0], indirgenmis_vector[:, 1], indirgenmis_vector[:, 2]) # x, y, z noktalarını çizdir

    for i, kelime in enumerate(kelimeler): # kelimeleri x, y, z noktalarının yanına yazma
        ax.text(indirgenmis_vector[i, 0], indirgenmis_vector[i, 1], indirgenmis_vector[i, 2], kelime, fontsize= 13)

    ax.set_title(baslik)
    ax.set_xlabel('Bileşen 1')
    ax.set_ylabel('Bileşen 2')
    ax.set_zlabel('Bileşen 3')
    plt.show()


plot_word_embeddings(word2vec_model, 'Word2Vec Gösterimi')
plot_word_embeddings(fasttext_model, 'FastText Gösterimi')