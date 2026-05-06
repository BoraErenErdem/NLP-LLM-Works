

"""

Amaç:
    - aynı kelimenin farklı bağlamlarda farklı anlamlarını bulmak
    - bunu yaparken lesk algoritmasını kullan

Yöntem: Lesk Algoritması
    - Lesk, bir kelimenin doğru anlamını belirlemek için bağlamdaki (cümledeki) diğer kelimeler ile karşılaştırma yapar.
    - WordNet sözlüğündeki tanımlar ile cümledeki kelimlere arasındaki ortak kelimeleri sayar.
    - En çok örtüşen anlam o kelimenin doğru anlamı seçilir.
    - örn. "bank" -> finansal anlam (banka) yada nehir kıyısı (river bank)

Adımlar:
    - nltk paketlerini indir
    - ilk cümle üzerinde bank kelimesini çözümle
    - ikinci cümle üzerinde bank kelimesini çözümle
    - her cümle için bulunan anlamı gör

pip install nltk, pywsd

"""


import nltk
nltk.download('wordnet') # wordnet sözlüğü
nltk.download('omw-1.4') # wordnet'in çoklu dil desteği
nltk.download('punkt') # tokenization için gerekli
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


sentence1 = "I go to the bank to deposit money"
target_word1 = 'bank'

sense1 = lesk(word_tokenize(sentence1), target_word1)
print(f'sentence: {sentence1}')
print(f'target word: {target_word1}')
print(f'predicted sense: {sense1.definition()}')


sentence2 = "The river bank is flooded after the heavy rain."
target_word2 = "bank"

sense2 = lesk(word_tokenize(sentence2), target_word2)
print(f'sentence: {sentence2}')
print(f'target word: {target_word2}')
print(f'predicted sense: {sense2.definition()}')