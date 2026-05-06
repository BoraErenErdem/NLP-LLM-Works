

"""

Amaç:
    - imdb film yorumları veri seti ile çalış
    - .csv dosyasından veri oku (imdb dataset.csv) (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    - text cleaning (küçük-büyük harf dönüşümü, özel karakterler ve noktalama işaretleri sil, stop wordsleri sil)
    - metinleri sayısal vektörlere dönüştür (bag of words)
    - kelime frekanslarının hesaplanması ve en sık geçen 5 kelimenin listelenmesi

pip install pandas
pip install scikit-learn

"""


import pandas as pd
import re # regex (veri temizleme)
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
nltk.download('stopwords')


df = pd.read_csv(r'C:\Projects\google_codes\metin_temsili\IMDB Dataset.csv')
print(df.head())

reviews = df['review'] # yorumlar
tags = df['sentiment'] # etiketler


def text_cleaning(text): # text cleaning fonksiyonu
    text = text.lower()
    text = re.sub(r"[^A-Za-z\s]", "", text) # harf ve boşluk dışındakileri sil
    text = " ".join([word for word in text.split() if len(word) > 2]) # 2 harften az olan kelimeleri sil
    # stop words
    stopwords_eng = set(stopwords.words('english'))
    text_list = text.split()
    text = [word for word in text_list if word not in stopwords_eng]
    return " ".join(text)


cleaning_reviews = [text_cleaning(i) for i in reviews] # her bir yorumu satır satır oluşturduğum metin temizleme fonksiyonuyla temizliyorum

cv = CountVectorizer()
comment_vectors = cv.fit_transform(cleaning_reviews[:75]) # ilk 75 yorumu sayısal vektöre çevirme
vocabulary = cv.get_feature_names_out() # vocabulary yani kelime sözlüğü
print(f'vocabulary: {vocabulary}')
vector_representation = comment_vectors.toarray() # vektör temsilleri
print(f'vector_representation: {vector_representation}')

df_bow = pd.DataFrame(vector_representation, columns=vocabulary) # vektör temsilleri dataframe'e dönüştürme
print(df_bow.head())

words_count = comment_vectors.sum(axis=0).A1 # her kelimenin toplam kaç adet geçtiğini bul  # A1 -> 1DArray'e dönüştürür..!
words_frequency = dict(zip(vocabulary, words_count)) # kelimeler ve frekansları dict içinde birleştirme
top_5_frequent_words = Counter(words_frequency).most_common(5) # Counter() ile en sık 5 kelimeyi bulma. burada most_common() kullandım çünkü dict olduğu için slicing olmuyor..! [:5]
print(f'top 5 frequent words: {top_5_frequent_words}')