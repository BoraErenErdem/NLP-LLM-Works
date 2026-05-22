

"""

Amaç:
    - BERT tabanlı soru cevap sistemi kurmak.
    - Model, verilen metinden belirli bir sorunun cevabını bulur.
    - Hugging Face'in "bert-large-uncased-whole-word-masking-finetuned-squad" adlı önceden eğitilmiş modelini kullan.
    - Model SQuAD (Stanford Question Answering Dataset) ile fine-tune edilmiştir.

Adımlar:
    - kütüphaneleri yükle
    - önceden eğitilmiş BERT modelini ve tokenizer'ını yükle
    - predict_answer fonksiyonu:
        - soruyu ve metni tokenize et
        - modeli çalıştır
        - en yüksek skorlu token'ı ara ve cevap olarak seç
        - tokenlar'ı string formuna çevir ve okunabilir hale getir
    - fonksiyonu örnek sorular üzerinde test et

pip install transformers, torch

"""


from transformers import BertForQuestionAnswering, BertTokenizer # BertForQuestionAnswering -> bert modelinin soru cevaplama versiyonu
import torch # pytorch kütüphanesi  # BertTokenizer -> BertTokenizer ile metinler bert modelinin anlayabileceği sayısal formata dönüştürür..!

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def predict_answer(context, question): # context -> metin # question -> soru
    encoding = tokenizer._encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True) # max_length= -> bert'in maximum token kapasitesi
    # ._encode_plus() -> metni modele uygun hale getiren fonksiyon (deprecated..!)
    input_ids = encoding['input_ids'] # input_ids -> her kelimeye karşılık gelen token_id'ler
    attention_mask = encoding['attention_mask'] # attention_mask -> padding tokenlarını maskeleyerek gerçek tokenlara odaklanılmasını sağlar
    with torch.no_grad(): # no_grad() -> eğitim yani backpropagation yok demek. # burada model çalıştırılır ve tahmin score'ları alınır
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
        # start_scores -> cevabın başladığı token olasılıkları    # end_scores -> cevabın bittiği token olasılıkları
    start_index = torch.argmax(start_scores, dim=1).item() # start_index -> en yüksek olasılıklı başlangıç token indexi
    end_index = torch.argmax(end_scores, dim=1).item() # end_index -> en yüksek olasılıklı bitiş token indexi
    # dim=1 -> pytorch'ta dim parametresi işlemin hangi eksende yapılacağını belirtir. burada dim=1 her batch örneği için token'lara bakıldı.
    # .item() -> tek elemanlı tensörü integer'a çevirir. örn. tensor([42]).item() -> 42
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1]) # input_ids içinden cevaba denk gelen token aralığını bulunur
    answer = tokenizer.convert_tokens_to_string(answer_tokens) # token'lar birleştirilerek okunabilir yazı haline getirildi
    return answer


question = 'What is machine learning?.'
context = 'Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.'
answer = predict_answer(context, question)
print(f'answer: {answer}')




# _encode_plus() deprecated olduğu için daha modern ve temiz yaklaşımla yaptım.
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def answer_model(question, context):
    encoding = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True) # tokenizer() içine call ettim
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask, return_dict=False)
    start_index = torch.argmax(start_scores, dim=1).item()
    end_index = torch.argmax(end_scores, dim=1).item()
    answer_tokens = input_ids[0][start_index: end_index + 1] # answer_tokens kısmını direkt tanımladım
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True) # answer kısmında tokenizer.decode() kullanarak içinde answer_tokens kısmını yazdım
    return answer


question = 'What is machine learning?.'
context = 'Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.'
answer = answer_model(question, context)
print(f'answer: {answer}')