

"""

Amaç:
    - kullanıcı-ürün puanlarından öğrenerek kullanıcıların ürünlere vereceği puanı tahmin eden sistem
    - yaklaşım: neural collaborative filtering en temel hali matrix factorization benzeri

Adımlar:
    - örnek veri seti oluştur (user_id, item_id, rating)
    - train test split
    - embedding tabanlı DNN modeli tanımla
    - modeli eğit ve test et
    - örnek kullanıcı-ürün üzerinden puan tahmini

pip install numpy, scikit-learn, tensorflow

"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Dot
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

np.random.seed(42) # numpy rastgelelik
tf.random.set_seed(42) # tensorflow rastgelelik


user_ids_all = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype='int32') # kullanıcı kimlikleri
item_ids_all = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 5], dtype='int32') # item kimlikleri
ratings_all = np.array([5, 4, 3, 2, 1, 4, 5, 3, 2, 1], dtype='float32') # gerçek puanlar

num_users = int(user_ids_all.max()) + 1 # kullanıcı sayısı (en büyük index + 1 sayısı kadar farklı kullanıcı sayısı)
num_items = int(item_ids_all.max()) + 1 # item sayısı (en büyük index + 1 sayısı kadar farklı item sayısı)

user_ids_train, user_ids_test, item_ids_train, item_ids_test, ratings_train, ratings_test = train_test_split(user_ids_all, item_ids_all, ratings_all, test_size=0.2, random_state=42)

def create_recommender_model(n_users, n_items, embedding_dim=8, lr=0.01): # n_users -> kullanıcı sayısı # n_items -> item sayısı # embedding_dim= -> embedding vektör boyutu
    user_input = Input(shape=(1,), name='user_input') # kullanıcı id girişi (her seferinde 1 kullanıcı id'si gireceğinden 1, yaptık..!)
    item_input = Input(shape=(1,), name='item_input') # item id girişi (her seferinde 1 item id'si gireceğinden 1, yaptık..!)
    user_embedding = Embedding(input_dim=n_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=n_items, output_dim=embedding_dim, name='item_embedding')(item_input)
    user_vector = Flatten(name='user_vector')(user_embedding)
    item_vector = Flatten(name='item_vector')(item_embedding)
    dot_score = Dot(axes=1, name='dot_user_item')([user_vector, item_vector]) # axes=1 -> 1. ekseni yani vektör boyutlarını kullanarak çarpıyor
    outputs = Dense(1, activation='linear', name='rating')(dot_score)
    model = Model(inputs=[user_input, item_input], outputs=outputs, name='recommender_model')
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model


embedding_dim = 8
lr = 0.01
model = create_recommender_model(num_users, num_items, embedding_dim, lr=lr)
history = model.fit([user_ids_train, item_ids_train], ratings_train, epochs=50, batch_size=4, validation_split=0.1, verbose=1)
test_loss, test_mae = model.evaluate([user_ids_test, item_ids_test], ratings_test, verbose=0)
print(f'test_loss: {test_loss}, test_mae: {test_mae}')



# region 1000 veri seti ile recommendation system
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Dot
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

np.random.seed(42)

n_users = 50
n_items = 100
n_ratings = 1000

print(user_ids_all.shape)
print(item_ids_all.shape)
print(ratings_all.shape)

n_users = int(user_ids_all.max()) + 1
print(n_users)
n_items = int(item_ids_all.max()) + 1
print(n_items)

user_ids_train, user_ids_test, item_ids_train, item_ids_test, ratings_train, ratings_test = train_test_split(user_ids_all, item_ids_all, ratings_all, test_size=0.2, random_state=42)

def recommender_model(n_user, n_item, embedding_dim=20, learning_rate=0.01):
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    user_embedding = Embedding(input_dim=n_user, output_dim=embedding_dim)(user_input)
    item_embedding = Embedding(input_dim=n_item, output_dim=embedding_dim)(item_input)
    user_vector = Flatten()(user_embedding)
    item_vector = Flatten()(item_embedding)
    dot_score = Dot(axes=1)([user_vector, item_vector])
    outputs = Dense(1, activation='linear')(dot_score)
    model = Model(inputs=[user_input, item_input], outputs=outputs, name='recommender_model')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

embedding_dim = 20
learning_rate = 0.01
model = recommender_model(n_users, n_items, embedding_dim, learning_rate=learning_rate)
history = model.fit([user_ids_train, item_ids_train], ratings_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
test_loss, test_mae = model.evaluate([user_ids_test, item_ids_test], ratings_test)
print(f'test_loss: {test_loss}, test_mae: {test_mae}')
# endregion