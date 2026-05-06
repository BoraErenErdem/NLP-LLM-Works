

"""

SMS spam veri seti üzerinden tf-idf ile analiz
Adımlar:
    - csv dosyasından sms verisi yükle (https://www.kaggle.com/datasets/hdza1991/sms-spam)
    - stopwords'leri temizle
    - TfidfVectorizer() ile sms verisi sayısal vektörlere dönüştür
    - her kelimenin ortalama tf-idf score'unu hesapla
    - sonuçları df'e aktar ve en yüksek score'a sahip 10 kelimeyi görüntüle

"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


df = pd.read_csv(r'C:\Projects\google_codes\metin_temsili\sms_spam.csv')
df.head()

print(df.columns)

text = df['text']
type = df['type']


def stopwords_cleaner(text):
    stopwords_eng = set(stopwords.words('english'))
    text_list = text.lower().split()
    text = " ".join([w for w in text_list if w not in stopwords_eng])
    return text

clean_text = [stopwords_cleaner(i) for i in text]

tf_idf = TfidfVectorizer()
text_vectors = tf_idf.fit_transform(clean_text)
vocabulary = tf_idf.get_feature_names_out()
print(f'vocabulary : {vocabulary}')
vector_representation = text_vectors.toarray()
print(f'vector_representation : {vector_representation}')

df_tf_idf = pd.DataFrame(vector_representation, columns=vocabulary)
print(df_tf_idf.head())

mean_tf_idf = df_tf_idf.mean(axis=0)
print(f'mean TF-IDF score: \n{mean_tf_idf.sort_values(ascending=False).head(10)}')