

"""

Amaç:
    - metindeki özel varlık isimlerini tespit etmek (kişi, yer, organizasyon vb.)
    - NER (named entity recognition) işlemini spacy kütüphanesi ile yap

Adımlar:
    - spacy ingilizce dil modelini yükle
    - metinde varlık ismi tanıma yap
    - bulunan varlıkları terminale print et

pip install spacy
python -m spacy download en_core_web_sm

"""

import pandas as pd
import spacy # NER için nlp kütüphanesi
nlp_model = spacy.load('en_core_web_sm') # spacy'nin ingilizce dil modelini yükler

sample_text = "Alice works at Amazon and live in London. She visited the British Museum last weekend."

doc = nlp_model(sample_text) # spacy modeli otomatik olarak tokenization, pos tagging, ner işlemlerini yapar..!

for i in doc.ents: # görselleştirebilmek için yaptım
    print(i.text, i.label_)


entity_list = [(i.text, i.label_, i.lemma_) for i in doc.ents]
df_entity = pd.DataFrame(entity_list, columns=['text', 'label', 'lemma'])
print(df_entity)
"""
                text   label               lemma
0               Alice  PERSON               Alice
1              Amazon     ORG              Amazon
2              London     GPE              London
3  the British Museum     ORG  the British Museum
4        last weekend    DATE        last weekend
"""