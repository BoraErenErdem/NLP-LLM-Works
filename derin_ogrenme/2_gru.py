

"""

Amaç: imdb film yorumları veri seti üzerinden GRU tabanlı sentiment classification modeli

Adımlar:
    - kütüphaneleri import et
    - veri setini yükle
    - padding işlemi uygula
    - GRU modeli kur
    - modeli compile et ve fit et
    - modeli test et (evaluation)
    - yeni yorum tahmini için custom function

pip install tensorflow, keras, pandas, numpy

"""


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GRU, Dropout, Embedding


num_words = 10000 # imdb veri setinde en çok kullanılan 10000 kelime (vocabulary'de tutulacak kelime sayısı)
max_sequence_length = 200 # her review 200 kelime ile sabitlersin (padding işlemi için)

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words) # X_train, X_test -> review   # y_train, y_test -> label (0 = negative, 1 = positive)
print(f'train shape: {X_train.shape}, test shape: {X_test.shape}')

X_train_padding = pad_sequences(X_train, maxlen=max_sequence_length)
X_test_padding = pad_sequences(X_test, maxlen=max_sequence_length)
print(f'X_train.shape: {X_train_padding.shape}, X_test.shape: {X_test_padding.shape}')

embedding_dim = 100 # kelimeler 100 boyutlu vektörlerle temsil edilir
# input_length= -> embedding katmanındaki her cümlenin kaç token olacağını belirtir. padding işlemi ile cümleler max_sequence_length=200 ile 200 token'a sabitlendi. !!deprecated!!
inputs = Input(shape=(max_sequence_length,))
embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)
X = GRU(64, return_sequences=False)(embedding)
X = Dropout(0.4)(X)
outputs = Dense(1, activation='sigmoid')(X)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train_padding, y_train, epochs=3, batch_size=128, validation_data=(X_test_padding, y_test), verbose=1)

test_loss, test_acc = model.evaluate(X_test_padding, y_test)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')


# custom function ile yeni cümle tahmini
# imdb dataset sayısal index kullandığından doğrudan kelimeler ile test edilemez. bunun için word_index alıp index_to_word mapping yapılması lazım.
word_index = imdb.get_word_index() # kelime'den index sözlüğü oluşturur
index_to_word = {v + 3: k for k, v in word_index.items()}
index_to_word[0] = '<PAD>' # padding
index_to_word[1] = '<START>' # cümlenin başlangıcı
index_to_word[2] = '<UNK>' # bilinmeyen cümle

def decode_review(encode_review):
    """
    sayısal review'ları tekrar kelimelere çevirir
    """
    return " ".join([index_to_word.get(i, '?') for i in encode_review])

def classify_review(review_sequence):
    """
    sayısal imdb yorumunu sınıflandırır
    """
    padded = pad_sequences([review_sequence], maxlen=max_sequence_length)
    prob = model.predict(padded)[0][0]
    label = 'positive' if prob > 0.5 else 'negative'
    return label, prob


decoded = decode_review(X_test_padding[0])
print(decoded)
pred_label, prob = classify_review(X_test_padding[0])
print(f'tahmin: {pred_label}, olasılık: {prob}')