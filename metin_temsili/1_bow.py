

"""

Amaç:
    - metin temsili (bag of words): metin listesi -> sayısal vektörler
    - sklearn CountVectorizer: kelimelerin kaç defa geçtiğini (frekansını) sayar ve vektör temsiline dönüştürür

Sonuç:
    - kelime kümesi (vocabulary)
    - her metin listesi sayısal vektörler ile temsil edilir

pip install scikit-learn

"""


# region CountVectorizer() ile bag of words (bow) yaparak kelime kümesi bulma (vocabulary) ve metin listesini sayısal vektörlere çevirme
from sklearn.feature_extraction.text import CountVectorizer

dokumanlar = [
    "kedi bahçede", # 1. cümle
    "kedi evde" # 2. cümle
]

cv = CountVectorizer()
dokuman_vektorleri = cv.fit_transform(dokumanlar) # dokumanlara bag of words (bow) uygulayarak sayısal vektörlere çevirme
kelime_kumesi = cv.get_feature_names_out() # CountVectorizer ile bulunan kelimelerin listesi (vocabulary)
print(f'kelime kümesi: {kelime_kumesi}')

vektor_temsili = dokuman_vektorleri.toarray() # dokuman_vektorleri'ni toarray() ile vektör temsili formatına çevirme
print(f'vektor temsili: {vektor_temsili}')
"""
vektor temsili: [[1 0 1] [0 1 1]]
"""
# endregion