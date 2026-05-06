

"""

Amaç:
    - RNN kullanarak duygu analizi (sentiment analysis)
    - Sınıflandırma problemi: restoran yorumları, etiket: positive, negative

Adımlar:
    - gerekli kütüphaneleri içe aktar (tensorflow.keras)
    - restoran review veri seti oluştur (gemini ile simülasyon verisi üret)
    - metin ön işleme (tokenization, padding, label encoding, train_test_split)
    - embedding: word2vec ile sayısal vektörler oluşturma
    - RNN modeli oluştur (embedding -> simpleRNN -> Dense layer output)
    - model compile ve fit
    - test seti üzerinde modeli değerlendir
    - user test (yeni cümlelerin sınıflandırılması için custom function tanımlama)

pip install, pandas, numpy, gensim, tensorflow, scikit-learn
"""


import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer # veriyi tokenize etmek için kullanılır
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding (doldurma) işlemi için kullanılır
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# restoran yorumları ve etiketleri
reviews_data = {
    "text": [
        "Yemekler harikaydı, her şey taze ve lezzetliydi.",
        "Garson çok ilgisizdi, siparişimi unuttular.",
        "Hayatımda yediğim en iyi pizzaydı, mutlaka deneyin.",
        "Mekan çok havasızdı, yemek yerken ter içinde kaldık.",
        "Mezeler efsane, özellikle atomu mutlaka isteyin.",
        "Fiyatlar porsiyonlara göre çok abartılı.",
        "Ambiyans büyüleyici, romantik bir akşam yemeği için ideal.",
        "Köfteler çok kuruydu, hiç beğenmedim.",
        "Servis hızı inanılmazdı, siparişimiz 10 dakikada geldi.",
        "Masalar birbirine çok yakın, mahremiyet sıfır.",
        "Personel güleryüzlü ve çok yardımcıydı.",
        "Tuvaletler kirliydi, hijyen konusunda sınıfta kaldılar.",
        "Tatlı menüsü çok zengin ve hepsi ev yapımı tadında.",
        "Çorba buz gibi geldi, geri gönderdik ama yine soğuktu.",
        "Manzarası harika, İstanbul ayaklarınızın altında.",
        "Gürültüden dolayı birbirimizi duymakta zorlandık.",
        "Kuzu tandır pamuk gibiydi, ağızda dağılıyordu.",
        "Rezervasyonumuz olmasına rağmen yarım saat bekletildik.",
        "Otopark sorunu yok, valeler çok kibar.",
        "İçecekler çok pahalı, bir suya bu kadar para verilmez.",
        "Vegan seçeneklerinin olması bizi çok mutlu etti.",
        "Makarna aşırı tuzluydu, yiyemedim.",
        "Sunumlar birer sanat eseri gibiydi.",
        "Müzik sesi o kadar yüksekti ki başımız ağrıdı.",
        "Peynir tabağı çok çeşitli ve kaliteliydi.",
        "Siparişlerin karışması tam bir fiyaskoydu.",
        "Ekmeği bile taptaze ve sıcacıktı.",
        "Balıklar taze değildi, bayat olduğu kokusundan belliydi.",
        "Dekorasyon çok şık ve modern tasarlanmış.",
        "Çay ikramı için teşekkür ederiz, ince bir düşünce.",
        "Hesapta gelmeyen şeyleri yazmışlar, dikkatli olun.",
        "Hamburger köftesi tam istediğim gibi az pişmişti.",
        "Salatadaki yeşillikler iyi yıkanmamıştı, kum geliyordu.",
        "Hafta sonu gitmemize rağmen servis aksamadı.",
        "Klima tam başımıza üflüyordu, rica etmemize rağmen kısmadılar.",
        "Çocuk oyun alanının olması ebeveynler için büyük kolaylık.",
        "Etin marine edilmesi başarılıydı ama porsiyon küçüktü.",
        "Mekan çok loş, menüyü okumakta zorlandık.",
        "Sıcak bir aile ortamı var, kendimizi evimizde hissettik.",
        "Garsonun tavrı çok kaba ve lakayttı.",
        "Tiramisu gerçek İtalyan tarifine sadık kalınarak yapılmış.",
        "Siparişi verdikten 45 dakika sonra yemek ancak geldi.",
        "Kahvaltı tabağı çok doyurucu ve ürünler doğaldı.",
        "Masadaki örtüler lekeliydi, değiştirilmesini istedik.",
        "Kokteyller yaratıcı ve dengeli aromalara sahipti.",
        "Dönerin içinden kıkırdak çıktı, çok kötü bir deneyimdi.",
        "Bahçesi çok huzurlu, şehir gürültüsünden uzak.",
        "Şarap menüsü yetersiz, çeşitlilik çok az.",
        "Girişte karşılanmamız çok nazikti.",
        "Piyasa fiyatlarının çok üzerinde, bir daha tercih etmem."
    ],
    "label": [
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative"]
}


# oluşturulan verileri df'e çevir
df = pd.DataFrame(reviews_data)
print(df.head())

# text preprocessing (tokenize, padding, label encoding, train_test_split)
tokenizer = Tokenizer() # tokenization ile kelimeler sayısal vektörlere çevrilir
tokenizer.fit_on_texts(df['text']) # vocabulary'i oluşturur ve vocabulary'nin her kelimesine index atanır
text_sequences = tokenizer.texts_to_sequences(df['text']) # vocabulary'deki kelimeleri index numarasına göre sayı dizisine çevirip sayı cümleleri oluşturuyor..!
word_index = tokenizer.word_index # daha net görmek için
print(word_index)

# padding
max_sequence_length = max(len(i) for i in text_sequences) # kelime bazında verideki max padding uzunluğu (9)
print(f'max_sequence_length: {max_sequence_length}')
X = pad_sequences(text_sequences, maxlen=max_sequence_length) # tüm cümleleri aynı uzunluğa getirir ve eksik kısımları 0 ile doldurur
print(f'X.shape: {X.shape}')

# label encoding
le = LabelEncoder()
y = le.fit_transform(df['label']) # positive = 1, negative = 0

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# embedding
sentences = [text.split() for text in df['text']]

# word2vec model
word2vec_model = Word2Vec(sentences=sentences, vector_size=50, window=5, min_count=1, sg=0, workers=4)
embedding_dim = 50 # her kelime 50 boyutlu vektörlerle temsil edilir (bunu tanımlamamın sebebi Embedding(output_dim=) parametresinde verebilmek için. zaten vector_size=50)

# embedding matrix -> word2vec ile önceden eğitilmiş vektörler var. bunları embedding katmanına aktarmak için embedding_matrix köprüsüne ihtiyaç var..!
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, idx in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[idx] = word2vec_model.wv[word] # her kelimenin Word2Vec vektörünü matrise yerleştir

# simpleRNN modeli
inputs = Input(shape=(max_sequence_length,))
x = Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(inputs)
# input_dim= -> kelime sayısı + 1   # output_dim= -> embedding_dim    # önceden eğitilmiş word to vec embedding_matrix   trainable= -> embedding ağırlıkları sabit kalır
x = SimpleRNN(units=50, return_sequences=False)(x) # units -> gizli katman sayısı   # return_sequences=False -> sadece son çıktıyı geri döndürür..!
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# compile ve fit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test), verbose=1)

# değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}; Test accuracy: {test_acc}')

# yeni cümlelerin sınıflandırılması için custom function
def classify_sentence(sentence): # yeni cümleyi alıp işleme sokar ve model ile sınıflandırmaya çalışır
    seq = tokenizer.texts_to_sequences([sentence]) # cümleyi sayısal vektörlere çevirir
    seq_padding = pad_sequences(seq, maxlen=max_sequence_length) # uzunluğu ayarlar

    predict = model.predict(seq_padding) # modelden predict ile olasılık tahmini yapar
    predict_class = (predict > 0.5).astype(int) # 0.5 üstü positive olarak sınıflandırılır

    if predict_class[0][0] == 1:
        label = 'positive'
    else:
        label = 'negative'
    return label

new_sentence = "Restoran çok temizdi ve yemekler çok güzeldi."
result = classify_sentence(new_sentence)
print(result)