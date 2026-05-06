

"""

Amaç:
    - cümledeki her kelimenin dilbilgisel türünü bulmak
    - spacy kullanarak her token için POS etiketini bul

Adımlar:
    - spacy ingilizce modeli yükle (en_core_web_sm)
    - cümle oluştur ve nlp modelinden geçir
    - her kelimenin pos etiketini yazdır

pip install spacy
python -m spacy download en_core_web_sm

"""


import spacy
nlp_model = spacy.load('en_core_web_sm')

sentence = "Can you recommend a good restaurant in London"

doc = nlp_model(sentence) # pos tagging

for i in doc:
    print(f'{i.text} {i.pos_}')

"""
Can AUX (yardımcı fiil)
you PRON (zamir)
recommend VERB (fiil)
a DET (belirteç)
good ADJ (sıfat)
restaurant NOUN (isim)
in ADP (edat)
London PROPN (özel ad)
"""