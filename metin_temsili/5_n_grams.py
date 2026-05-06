

"""

Verilen örnek cümleler üzerinden n-gram (unigram, bigram, trigram) analizi
Adımlar:
    - örnek belgeleri tanımla
    - CountVectorizer() ile unigram, bigram, trigram örnekleri
    - her model için özellik (kelime veya kelime grubu) çıkart

pip install scikit-learn

"""


from sklearn.feature_extraction.text import CountVectorizer

belgeler = [
    "Bu çalışma bir NGRAM çalışmasıdır.",
    "Bu çalışma doğal dil işleme çalışmasıdır."
]

unigram = CountVectorizer(ngram_range=(1, 1)) # n = 1  # ngram_range= -> N-Gram modelini belirler
bigram = CountVectorizer(ngram_range=(2, 2)) # n = 2
trigram = CountVectorizer(ngram_range=(3, 3)) # n = 3

X_unigram = unigram.fit_transform(belgeler) # unigram ile belgeler'i sayısal vektöre dönüştürme
unigram_features = unigram.get_feature_names_out() # unigram özellikleri
print(f'unigram özellikleri: {unigram_features}')

X_bigram = bigram.fit_transform(belgeler)
bigram_features = bigram.get_feature_names_out() # bigram özellikleri
print(f'bigram özellikleri: {bigram_features}')

X_trigram = trigram.fit_transform(belgeler)
trigram_features = trigram.get_feature_names_out() # trigram özellikleri
print(f'trigram özellikleri: {trigram_features}')

print(f'unigram: {unigram_features}')
print(f'bigram: {bigram_features}')
print(f'trigram: {trigram_features}')

"""
unigram: ['bir' 'bu' 'dil' 'doğal' 'işleme' 'ngram' 'çalışma' 'çalışmasıdır']
bigram: ['bir ngram' 'bu çalışma' 'dil işleme' 'doğal dil' 'işleme çalışmasıdır' 'ngram çalışmasıdır' 'çalışma bir' 'çalışma doğal']
trigram: ['bir ngram çalışmasıdır' 'bu çalışma bir' 'bu çalışma doğal' 'dil işleme çalışmasıdır' 'doğal dil işleme' 'çalışma bir ngram' 'çalışma doğal dil']
"""