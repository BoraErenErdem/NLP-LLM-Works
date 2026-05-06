

"""

Amaç:
    - spacy kullanarak cümlenin her kelimesini (yani tokenını) incelemek
    - her kelimenin gövde (lemma), POS (dilbilgisel kategorisi), morfoloji, çoğul olup olmadığı vb. bilgilerin çıkarılması

Adımlar:
    - spacy ingilizce modeli yükle (en_core_web_sm)
    - örnek cümle üzerinden nlp işlemleri uygula
    - her kelimenin özelliklerini ekrana yazdır

pip install spacy
python -m spacy download en_core_web_sm

"""


import spacy
nlp_model = spacy.load('en_core_web_sm')

sentence = "They are playing football in the parks"

doc = nlp_model(sentence) # nlp işleminden geçerek tokenization, pos tagging, dependency parse, lemma, morfoloji yapar.

for i in doc:
    print(f'text -> {i.text}') # kelimenin kendisi
    print(f'lemma -> {i.lemma_}') # kök (gövde yani lemma) hali
    print(f'pos -> {i.pos_}') # genel dilbilgisel kategorisi
    print(f'tag -> {i.tag_}') # daha detaylı pos etiketi
    print(f'dependency -> {i.dep_}') # cümledeki sözdizimsel rolü (subject, object vb.)
    print(f'shape -> {i.shape_}') # kelimenin karakter yapısı (Xxx, ddd vb.)
    print(f'is alpha -> {i.is_alpha}') # kelime sadece harflerden mi oluşuyor
    print(f'is stop -> {i.is_stop}') # kelime stopwords mü değil mi
    print(f'morfoloji -> {i.morph}') # morfolojik özellik
    print(f'is plural -> {'Number=Plur' in i.morph}') # kelime çoğul mu değil mi
    print('-'*50)


"""
text -> They
lemma -> they
pos -> PRON
tag -> PRP
dependency -> nsubj
shape -> Xxxx
is alpha -> True
is stop -> True
morfoloji -> Case=Nom|Number=Plur|Person=3|PronType=Prs
is plural -> True
--------------------------------------------------
text -> are
lemma -> be
pos -> AUX
tag -> VBP
dependency -> aux
shape -> xxx
is alpha -> True
is stop -> True
morfoloji -> Mood=Ind|Tense=Pres|VerbForm=Fin
is plural -> False
--------------------------------------------------
text -> playing
lemma -> play
pos -> VERB
tag -> VBG
dependency -> ROOT
shape -> xxxx
is alpha -> True
is stop -> False
morfoloji -> Aspect=Prog|Tense=Pres|VerbForm=Part
is plural -> False
--------------------------------------------------
text -> football
lemma -> football
pos -> NOUN
tag -> NN
dependency -> dobj
shape -> xxxx
is alpha -> True
is stop -> False
morfoloji -> Number=Sing
is plural -> False
--------------------------------------------------
text -> in
lemma -> in
pos -> ADP
tag -> IN
dependency -> prep
shape -> xx
is alpha -> True
is stop -> True
morfoloji -> 
is plural -> False
--------------------------------------------------
text -> the
lemma -> the
pos -> DET
tag -> DT
dependency -> det
shape -> xxx
is alpha -> True
is stop -> True
morfoloji -> Definite=Def|PronType=Art
is plural -> False
--------------------------------------------------
text -> parks
lemma -> park
pos -> NOUN
tag -> NNS
dependency -> pobj
shape -> xxxx
is alpha -> True
is stop -> False
morfoloji -> Number=Plur
is plural -> True
"""