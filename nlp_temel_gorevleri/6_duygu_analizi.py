

"""

Amaç:
    - Amazon yorumlarının positive mi negative mi olduğunu sınıflandırmak
    - binary classification problemi
    - dataset -> https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv

Adımlar:
    - text cleaning ve preprocessing (tokenization, stopwords, lemmatization)
    - duygu analizi (Sentiment Intensity Analizer - VADER)
    - tahmin sonuçlarını confusion matrix ve classification report ile değerlendir

!!! DNN (LSTM vb.) + Embedding işlemi ile çok daha doğru ve stabil sonuç elde edilir..!

pip install scikit-learn, pandas, nltk

"""

# region VADER kullanarak sentiment analysis
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon') # vader sözlüğü
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer # duygu analizi için
from sklearn.metrics import confusion_matrix, classification_report



url = 'https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv'
df = pd.read_csv(url)
print(df.head())

wnl = WordNetLemmatizer()

def clean_and_preprocess(text):
    text = word_tokenize(text.lower()) # küçük harfe çevir ve tokenize et
    stopwords_eng = stopwords.words('english')
    text = " ".join([i for i in text if i not in stopwords_eng])
    text = " ".join([wnl.lemmatize(i) for i in text.split()])
    return text

# df['clean_review'] = [clean_and_preprocess(i) for i in df['reviewText']]
df['clean_review'] = df['reviewText'].apply(clean_and_preprocess)
print(df.head())

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text): #
    score = analyzer.polarity_scores(text) # vader puanlarını al
    sentiment = 1 if score['pos'] > 0 else 0
    return sentiment

df['predicted_sentiment'] = df['clean_review'].apply(analyze_sentiment)
print(df.head())

print(f'confusion matrix: {confusion_matrix(df['Positive'], df['predicted_sentiment'])}')
print(f'classification report: {classification_report(df['Positive'], df['predicted_sentiment'])}')
# endregion



# region LSTM + Embedding ile sentiment analysis
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


url = 'https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv'
df = pd.read_csv(url)
print(df.head())
print(df.columns)

wnl = WordNetLemmatizer()

def clean_and_preprocess(text):
    text = text.lower()
    text = word_tokenize(text)
    stopwords_eng = stopwords.words('english')
    text = " ".join([i for i in text if i not in stopwords_eng])
    text = " ".join([wnl.lemmatize(i) for i in text.split()])
    return text

# df['clean_review'] = [clean_and_preprocess(i) for i in df['reviewText']]
df['clean_review'] = df['reviewText'].apply(clean_and_preprocess)
print(df.head())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_review'])
text_sequences = tokenizer.texts_to_sequences(df['clean_review'])
word_index = tokenizer.word_index
print(word_index)

max_sequence_length = max(len(i) for i in text_sequences)
print(max_sequence_length)
X = pad_sequences(text_sequences, max_sequence_length)
print(X.shape)
y = df['Positive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

sentences = [text.split() for text in df['clean_review']] # Word2Vec modeli için veriyi liste içinde liste şekline hazırlıyor..!
word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=0, workers=4)
embedding_dim = 100

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim)) # boş matrix oluşturur -> her kelime için Word2Vec vektörünü bulur -> matrixe yerleştirir -> embedding katmanına verir
for word, idx in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[idx] = word2vec_model.wv[word]

inputs = Input(shape=(max_sequence_length,))
embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(inputs)
X = LSTM(100, return_sequences=False)(embedding)
X = Dropout(0.2)(X)
outputs = Dense(1, activation='sigmoid')(X)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}. Test Accuracy: {test_acc}')
pred = (model.predict(X_test) > 0.5).astype(int)
print(f'confusion matrix: {confusion_matrix(y_test, pred)}')
print(f'classification report: {classification_report(y_test, pred)}')
# endregion