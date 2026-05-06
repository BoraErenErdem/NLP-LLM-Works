

"""

Amaç:
    Temel veri temizleme adımları:
        - fazla boşlukların kaldırılması
        - büyük harflerin küçük harflere dönüştürülmesi
        - noktalama işaretlerinin kaldırılması
        - özel karakterlerin kaldırılması
        - yazım hatalarının düzeltilmesi
        - HTML ve URL etiketlerinin kaldırılması

pip install textblob
pip install beautifulsoup4

"""

# region fazla boşlukların temizlenmesi
raw_text = "Python,    Google    NLP    dersi."
print(raw_text.split()) # split() -> metni sadece boşluklara göre bölerek ayırır ve listeye çevirir. argüman yoksa tek boşluk olarak ayırır.
normalized_text_1 = " ".join(raw_text.split()) # " ".join() -> listedeki elemanların aralarına tek boşluk koyarak tekrar birleştirir.
print(f'fazla boşlukları temizle: {normalized_text_1}')
# endregion

# region büyük ve küçük harf dönüşümü
raw_text = "PYTHON, GooGle NLP"
normalized_text_2 = raw_text.lower() # .lower() -> tüm harfleri küçük harf yapar.
print(f'temizlenmiş veri: {normalized_text_2}')
# endregion

# region noktalama işaretlerinin temizlenmesi
import string
raw_text = "AI Natural-Language-Processing!"
normalized_text_3 = raw_text.translate(str.maketrans("","", string.punctuation)) # noktalama işaretlerini siler
print(f'noktalama işaretleri temizlenmesi: {normalized_text_3}')
# endregion

# region özel karakterlerin temizlenmesi
import re # regex düzenli ifadeler kütüphanesi
raw_text = "Natural @ Language % Processing"
normalized_text_4 = re.sub(r"[^A-Za-z0-9\s]", "", raw_text) # A-Z'ye, a-z'ye ve 0-9'a kadar olanları ve boşluk karakterini koru, bunlar haricini sil..!
normalized_text_4 = " ".join(normalized_text_4.split())
print(f'özel karakterlerin ve gereksiz boşlukların temizlenmesi: {normalized_text_4}')
# endregion

# region yazım hatalarının düzeltilmesi
from textblob import TextBlob
raw_text = "It is amazig in 2045"
normalized_text_5 = TextBlob(raw_text).correct() # .correct() -> yazım hatalarını düzeltir.
print(f'yazım hatalarının düzeltilmesi: {normalized_text_5}')
# endregion

# region html ve url etiketlerinden düz metin elde etme
from bs4 import BeautifulSoup
raw_html = "<div> 2045 Google </div>"
normalized_text_6 = BeautifulSoup(raw_html, 'html.parser').get_text() # html etiketlerinden temizlenmiş şekilde cümleyi alır.
print(f'html etiketlerinden temizlenmiş metin: {normalized_text_6}')
# endregion