

"""

Amaç:
    - Hugging Face'in Transformers kütüphanesini kullanarak metin özetleme (text summarization) pipeline'ı oluşturmak ve verilen uzun metni kısa bir özet haline getirmek.
    - model önceden eğitilmiş büyül dil modeli olucak. metnin ana fikrini koruyarak kısa özet üret.

Adımlar:
    - gerekli kütüphaneleri yükle (Transformers, pytorch)
    - özetleme yani summarization pipeline'ı yükle
    - uzun metin tanımla
    - modeli çalıştır ve özet oluştur
    - sonucu ekrana yazdır

pip install transformers, pytorch

"""


from transformers import BartForConditionalGeneration, BartTokenizer # BartTokenizer -> metni BART modelinin anlayabileceği tokenlara çevirir
# BartForConditionalGeneration -> BART'ın encoder + decoder mimarisini kullanarak özetleme/çeviri modellerini yükler.

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn') # .from_pretrained() -> hugging face hub'da önceden eğitilmiş modeli veya tokenizer'ı indirip yükler..!
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

text = """
As the giant minute hand of the clock tower edged toward midnight, Alper stopped in front of the old antiquarian shop in the town square. Rain made the cobblestone streets glisten, and the flickering yellow light of the streetlamps refracted in the puddles. Alper was an antiquarian in his mid-thirties, lost in the routine cycle of life. However, tonight, the object sitting in the shop's display window was enough to disrupt that entire routine: a palm-sized antique pocket watch with intricate gears and brass engravings.
The shop’s elderly owner was arranging the shelves. Alper jingled the bell on the door and stepped inside. As water dripped from his raincoat, forming small pools on the floor, he headed straight for the display case.
"Do you know what that is?" the old man asked without turning around. His voice carried the scent of old, dusty libraries.
"It just... looks mesmerizing," Alper said, unable to take his eyes off the watch. "Is it for sale?"
The old man turned slowly, a faint, ambiguous smile appearing on his deeply wrinkled face. "That watch is not for sale, my boy. It chooses. But beware, it does not show time. It bends it. They call it the Clock of Regrets. It grants you the return of a single moment in your life that you wish to change. But the price is heavy; stolen time always demands its own interest."
Alper dismissed the old man's words as an old man's attempt to create an air of mystery. Suddenly, the deep, aching regret he had carried inside for years awakened. Five years ago, he had failed to stop his sister, Elif, from driving off into that stormy night. They had fought, and Elif had driven away in anger, never to return.
"I’ll pay whatever you want," Alper said, his voice trembling.
The old man took the watch from the shelf and placed it in Alper’s palm. "The price is not money," he whispered. "It is the days deducted from your future. Do you accept?"
Without hesitation, Alper nodded. The brass body of the watch was ice-cold, but suddenly its gears began to spin furiously, making a wild, frantic sound just as the town clock tower's bells began to toll midnight.
When he opened his eyes, Alper found himself in that familiar living room from five years ago. Logs were crackling in the fireplace, and outside, a heavy rain was pouring, just like now. The clatter of dishes from the kitchen filled his ears.
His heart was pounding as if it would burst from his chest. He rushed into the kitchen. Elif was there. Alive, breathing, reaching for the car keys on the counter. Her face bore the same hurt and angry expression from that day.
"I hate you, Alper!" Elif shouted, her eyes welling with tears. "I won't stay in this house for another minute!"
Alper remembered the lines from that moment in the past. Normally, he would have yelled back, provoking her and letting her slam the door and leave. But this time, he didn't. He practically collapsed at Elif's feet, gripped the keys tightly, and wrapped his arms around his sister.
"I'm sorry," Alper sobbed. "I'm so sorry, Elif. You're right. It's all my fault. Please don't go. Just stay here tonight, please."
Elif froze. Faced with her brother’s sudden and intense surrender, her anger instantly turned to confusion, then to tenderness. She dropped the keys and hugged her brother back. "Okay," she said in a low voice. "I'm not going."
Alper felt the colossal weight inside him lift like a bird. He had changed time. His sister was going to live. He had succeeded.
But right at that moment, the metallic screeching of the antique watch's frantically spinning gears began to ring in his ears. The gears were turning so fast that a burning heat bloomed against Alper’s chest from the friction. The world around him slowly began to blur, and colors bled into one another.
When Alper opened his eyes again, he was back in front of the antiquarian shop in the town square. The rain was still falling. He immediately checked his pocket; the watch was there, but it had completely stopped, its gears lifeless.
He quickly pulled out his phone and dialed Elif’s number with trembling fingers. The phone rang, rang, and then connected.
"Alper? What's wrong at midnight, bro? Is there a problem?" came Elif’s cheerful voice.
Tears streamed down Alper’s face. "No, no, there’s no problem at all. I just wanted to hear your voice. I love you so much."
"I love you too, dummy. Go to sleep now," Elif laughed and hung up.
Alper stood under the streetlamp, crying tears of pure joy. He had changed the past, and his sister was alive. Whatever that "heavy price" the old man spoke of was, it was nothing compared to Elif being alive. He turned toward the shop window to look at his reflection in the glass.
But what he saw made him freeze.
The person staring back from the window's reflection was not a young man in his mid-thirties. It was an old man in his eighties, his hair completely white, his face etched with deep wrinkles, and the light faded from his eyes. In shock, Alper looked at his hands; his skin was spotted, his fingers trembling. In disbelief, he touched the old man in the mirror.
The door of the antiquarian shop slowly swung open. The old man from inside stepped out and placed his hand on Alper’s shoulder. As Alper looked closely at the old man now, he realized with horror how much the man’s facial features actually resembled his own.
"I told you, my boy," the antiquarian said, his voice carrying a profound tenderness. "Time always collects the interest on stolen days. To give your sister those long years she deserved, you sacrificed all the years of your own future. You grew old in her place."
With trembling hands, Alper pulled the stopped watch from his pocket. Was he sorry? No. Elif was alive, perhaps married with children, living a happy life. He had traded his own youth for her life.
"What happens now?" Alper asked, his voice now sounding old and weary.
The antiquarian smiled and opened the shop door wide. "Now, it is your turn to take over the shop. Because the story of those who bend time never ends. One day, another regretful soul will walk through this door... And you will hand him the watch."
With heavy steps, Alper walked into the shop. Amidst the thousands of old books on the shelves and the rhythmic ticking of the clocks, he took his first step into the most peaceful—and oldest—period of his life. Time was ticking away, but he carried the proud serenity of having conquered time for the sake of love.
"""

inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True) # return_tensors= 'pt' -> tokenları pytorch tensor formatında döndürür.
# max_length= -> girdinin maximum token uzunluğu   # truncation=True -> girdi max_length (max token) sayısını sayısını aşarsa keser.

summary_ids = model.generate(inputs['input_ids'], max_length=500, min_length=5, do_sample=True) # model token_id'lerini alıp özet token_id'leri üretir.
# inputs['input_ids'] -> tokenizer'dan gelen token_id'ler   # max_length= -> üretilecek özetin maximum token uzunluğu   # min_length= -> üretilecek özetin minimum token uzunluğu #
# do_sample=True -> rastgelelik ekleyerek modelin farklı özetler üretmesini sağlar (True/False)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True) # token_id'ler tekrar metne çevrilir.
# summary_ids[0] -> ilk özet dizi   # skip_special_tokens=True -> modelin eğitim sırasında kullandığı özel tokenları atlar (<s>, </s>, <pad> vb.)

print(summary)
"""
metin
→ tokenizer() → token ID'leri
→ model.generate() → özet token ID'leri
→ tokenizer.decode() → özet metin
"""


#from transformers import pipeline # pipeline -> hugging face pipline ile önceden eğitilmiş modeller kullanılabilir
# eğer eski transformers versiyonu olsaydı pipeline içine direkt 'summarization' diyerek özetleme işlemi yapabilirdim. yeni sürümde summarization hariç diğer sürümler destekleniyor..!
#summarizer = pipeline('summarization') # 'summarization' -> model metin özetleme pipeline'ı ve arkada belirlenen küçük llm modelini çağırılır.
# pipeline() -> içine yazılan görevin adı ve modelin ne yapacağını belirler. her görev için hugging face uygun varsayılan modeli yükler.
# 'summarization' -> metin özetleme  # 'sentiment-analysis' -> duygu analizi  # 'text-generation' -> metin üretimi  # 'translation' -> çeviri  # 'question-answering' -> soru cevaplama
# 'ner' -> isimli varlık tanıma  # 'fill-mask' -> eksik kelime tamamlama
#summary = summarizer(text, max_length=20, min_length=5, do_sample=True) # summarizer() fonksiyonu liste return eder ve her öge dict yapısındadır.
# max_length= -> özetin maximum token uzunluğunu belirler
# min_length= -> özetin en az kaç token uzunluğunda olacağını belirtir
# do_sample=True -> rastgelelik ekleyerek modelin farklı özetler üretmesini sağlar (True/False)
#print(summary[0]['summary_text'])