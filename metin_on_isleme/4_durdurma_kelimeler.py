

"""

stop words çıkarma yöntemleri
    - ingilizce stop words çıkarma (nltk)
    - türkçe stop words çıkarma (nltk)
    - manuel olarak stop words çıkarma

pip install nltk

"""

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') # stop words kelime veri setini indirir

# region ingilizce stop words çıkarma
stop_words_eng = set(stopwords.words('english')) # ingilizce stop words listesi
eng_text = "This is just a simple example to show how stop words can be removed from sentences"
eng_text_list = eng_text.split()
print(eng_text_list)
filtered_words_eng = [word for word in eng_text_list if word.lower() not in stop_words_eng] # ingilzice stop wordsler temizlenir
print(f'original: {eng_text}')
print(f'filtered: {filtered_words_eng}')
# endregion

# region türkçe stop words çıkarma
stop_words_tr = set(stopwords.words('turkish')) # türkçe stop words listesi
tr_text = "Merhaba, bugün sizler ile birlikte Google NLP eğitimi gerçekleştiriyoruz. Bu eğitim sizler için çok faydalı olacaktır."
tr_text_list = tr_text.split()
print(tr_text_list)
filtred_words_tr = [word for word in tr_text_list if word.lower() not in stop_words_tr] # türkçe stop wordsler temizlenir
print(f'original: {tr_text}')
print(f'filtered: {filtred_words_tr}')
# endregion

# region manuel stop words çıkarma işlemi
custom_tr_stopwords = ["bu", "ile", "de", "da", "mi", "ki"]
custom_text = "Bu bir denemedir, bunun için amacımız metinlerde ki bazı kelimeleri çıkartmak."
custom_text_list = custom_text.split()
filtered_custom_words_tr = [w for w in custom_text_list if w.lower() not in custom_tr_stopwords]
print(filtered_custom_words_tr)
# endregion