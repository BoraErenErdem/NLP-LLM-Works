

"""

Amaç:
    - sms spam collection veri setini kullanarak spam ve normal mesaj şeklinde sınıflandırma yap
    - bu problem binary classification problemi; decision tree (karar ağacı) algoritması ile çöz

Adımlar:
    - veri setini yükle
    - metin ön işleme (temizlik + lower + stopwords + lemmatization (gövde))
    - feature extraction (özellik çıkarımı) için bag of words algoritması kullan (CountVectorizer())
    - modeli eğit
    - evaluation yap; confusion matrix + accuracy

pip install scikit-learn, pandas, nltk

"""


import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # gövde bulma (lemmatization)
nltk.download('wordnet') # lemmatization için wordnet indirir
nltk.download('omw-1.4') # farklı dil desteği sağlar


df = pd.read_csv(r'C:\Projects\google_codes\metin_temsili\sms_spam.csv')
print(df.head())
print(df.columns)

df.columns = ['label', 'text'] # 'type' sütun ismini 'label' olarak değiştirdim
print(df.columns)
print(f'label: {df['label'].value_counts()}')
print(f'text: {df['text'].value_counts().sum()}')
print(df.isnull().sum()) # eksik değer kontrolü

wnl = WordNetLemmatizer()

def text_cleaning(text):
    text = re.sub(r'[^A-Za-z]', " ", text)
    text = text.lower()
    stopwords_eng = stopwords.words('english')
    text = " ".join([i for i in text.split() if i not in stopwords_eng])
    text = " ".join([wnl.lemmatize(i) for i in text.split()]) # gövde bulma (lemmatization)
    return text


# df['clean_text'] = [text_cleaning(i) for i in df['text']] # bu da list comprehension ile temizlenen veriyi df'e ekler
df['clean_text'] = df['text'].apply(text_cleaning) # temizlenen veriyi df'e ekleme
print(df.head())

X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train) # train verisi
X_test_cv = cv.transform(X_test) # test verisi

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_cv, y_train)
pred = dt_model.predict(X_test_cv)
print(f'decision tree model accuracy: {accuracy_score(y_test, pred)}, confusion matrix: {confusion_matrix(y_test, pred)}')