

"""

NLP için temel ön adımlardan olan tokenizasyon yapılıcak.
    - kelime tokenizasyonu
    - cümle tokenizasyonu

pip install nltk

"""

import nltk
nltk.download('punkt') # kelime ve cümle tokenizasyonu için gerekli veri. # split() ile tokenizasyon yapılmaz çünkü punkt gibi akıllı şekilde tokenlara ayıramaz..!

# örnek text
raw_text = "Merhaba Google! Bu bir NLP eğitimidir. NLP eğitiminin ilerleyen aşamalarında LLM konusunu öğrenelim."

# region kelime tokenizasyonu
word_tokens = nltk.word_tokenize(raw_text) # .word_tokenize() -> kelime tokenizasyonu yapar.
print(f'kelime tokenları: {word_tokens}')
# endregion

# region cümle tokenizasyonu
sentence = nltk.sent_tokenize(raw_text) # .sent_tokenize() -> cümle tokenizasyonu yapar.
print(f'cümle tokenları: {sentence}')
# endregion