

"""

Kök (stemming) ve gövde (lemmatization) bulma
    - stemming: porter stemmer
    - lemmatization: word net lemmatizer

pip install nltk

"""

import nltk
nltk.download('wordnet') # gövde (lemma) bulmak için gerekli wordnet veri tabanı
nltk.download('omw-1.4') # wordnet veri tabanı için ek dil desteği

# region kök (stemming)
from nltk.stem import PorterStemmer # ingilizce için popüler stemmer algoritması
stemmer = PorterStemmer() # porter stemmer nesnesi oluştur
words_stem = ["playing", "played", "plays", "happier", "happily", "studying", "studies"]
stems = [stemmer.stem(w) for w in words_stem]
print(f'original: {words_stem}')
print(f'kökler (stems): {stems}')
# endregion

# region gövde (lemmatization)     # lemmatization işlemlerinde pos= parametresi varsayılan olarak (n) isim'dir. her kelimenin türüne göre pos= parametresini belirlemek gerekir..!
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words_lemma = ['running', 'ran', 'gone', 'better', 'children']
lemma = [lemmatizer.lemmatize(w) for w in words_lemma]
print(f'original: {words_lemma}')
print(f'gövdeler (lemma): {lemma}')
# endregion