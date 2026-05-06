

"""

örnek cümleler üzerinden TF-IDF uygulayarak cümleleri vektörleştirmek.
Adımlar:
    - belge oluştur
    - TF-IDF Vectorizer ile belgeleri sayısal vektörlere dönüştür
    - kelime kümesi (vocabulary) çıkar
    - belgelerin tf-idf vektör temsillerini bul
    - tüm belgeler için kelimelerin ortalama TF-IDF değerlerini hesapla

pip install pandas
pip install scikit-learn
pip install numpy

"""


# region TfidfVectorizer() ile belgelerin tf-idf vektör temsillerini bulup tüm belgelerin ortalama tf-idf değerlerini hesaplayarak cümleleri vektörlerştirme
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

belgeler = [
    "Köpek çok tatlı bir hayvandır", # 1.belge
    "Köpek ve kuşlar çok tatlı hayvanlardır", # 2.belge
    "inekler süt üretirler" # 3.belge
]

tf_idf = TfidfVectorizer()
belge_vektorleri = tf_idf.fit_transform(belgeler) # belgelere TfidfVectorizer() uygulayarak sayısal vektörlere çevirme
kelime_kumesi = tf_idf.get_feature_names_out() # TfidfVectorizer() ile oluşan kelime kümesini (vocabulary) bulma
print(f'Kelime kümesi: {kelime_kumesi}')

vektor_temsili = belge_vektorleri.toarray() # belge_vektorleri'ni .toarray() ile vektor temsili formatına çevirme
print(f'vektör temsilleri: {vektor_temsili}')
"""
[[0.51741994 0.51741994 0.         0.         0.3935112  0.         0.         0.3935112  0.         0.3935112  0.]
    [0.         0.         0.45954803 0.45954803 0.34949812 0.         0.         0.34949812 0.45954803 0.34949812 0.]
    [0.         0.         0.         0.         0.         0.57735027        0.57735027 0.         0.         0.         0.57735027]]
"""

df_tf_idf = pd.DataFrame(vektor_temsili, columns=kelime_kumesi) # daha iyi görebilmek için df'e çevirdim
print(df_tf_idf) # burada her kelimenin TF-IDF değeri görünebilir
"""
        bir  hayvandır  hayvanlardır  ...        ve       çok  üretirler
0  0.51742    0.51742      0.000000  ...  0.000000  0.393511    0.00000
1  0.00000    0.00000      0.459548  ...  0.459548  0.349498    0.00000
2  0.00000    0.00000      0.000000  ...  0.000000  0.000000    0.57735
"""

ortalama_tf_idf = df_tf_idf.mean(axis=0) # her kelimenin belgeler arasındaki ortalama tf_idf değerinin hesaplanması
print(f'kelimelerin ortalama tf_idf değerleri: \n{ortalama_tf_idf}')
"""
kelimelerin ortalama tf_idf değerleri: 
bir             0.172473
hayvandır       0.172473
hayvanlardır    0.153183
inekler         0.192450
kuşlar          0.153183
köpek           0.247670
süt             0.192450
tatlı           0.247670
ve              0.153183
çok             0.247670
üretirler       0.192450
"""
# endregion