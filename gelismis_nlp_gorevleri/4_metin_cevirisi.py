

"""

Amaç:
    - hugging face'in MarianMT (Marian Machine Translation) modelini kullanarak çeviri yapmak
    - ingizilceden fransızcaya çeviri yap
    - Model, Helsinki NLP grubu tarafından geliştirilen bir model. bu modeller çok sayıda dil çifti arasından çeviri yapmak üzere eğitildi.

Adımlar:
    - kütüphaneleri yükle
    - model adını belirle
    - tokenizer ve modeli yükle
    - girdi metnini tanımla
    - tokenizer ile metni modelin anlayabileceği formata çevir
    - model ile çeviri işlemini yap
    - token_id'leri tekrar metne dönüştür ve sonucu ekrana yazdır

pip install transformers, torch
"""


from transformers import MarianMTModel, MarianTokenizer # çeviri modeli ve uyumlu tokenizer

model_name = 'Helsinki-NLP/opus-mt-en-fr' # hugging face'te önceden eğitilmiş hazır model

tokenizer = MarianTokenizer.from_pretrained(model_name) # metni sayısal token_id'lere dönüştürür
model = MarianMTModel.from_pretrained(model_name) # token_id'leri alır ve hedef dile çevrilmiş token_id'ler üretir

text = "Hello, what is your name?"

inputs = tokenizer(text, return_tensors='pt', padding=True)
translate_tokens = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']) # çeviri çıktısını token_id formatında üretir
translate_text = tokenizer.decode(translate_tokens[0], skip_special_tokens=True) # çevrilmiş token_id'ler okunabilir metne çevrildi
print(f'translated_text: {translate_text}')