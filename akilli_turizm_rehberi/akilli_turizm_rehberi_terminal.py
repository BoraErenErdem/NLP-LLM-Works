

"""
Problem Tanımı:
    - Akıllı Turizm Rehberi (LLM): Kullanıcı Türkiye turizmi ile ilgili sorular sorar. (örn. Antalya'da neler gezinir ne yenir vb)
    - LLM akıllı şekilde doğal dilde cevap üretir.
    - Google Gemma3 (açık kaynak) modelini Ollama frameworkü üzerinden kullan.
    - Bu proje için Streamlit ile kullanıcı arayüzü (UI) oluştur.

Model Tanıtımı:
    - Google Gemma3: Google'ın geliştirdiği açık kaynak LLM modelidir. 4b parametreli modelini kullan.
    - Local'de çalışır yani on-prem olarak çalışır. Bu sayede bilgiler bulut ortamına gitmez.

Kütüphaneler:
    - pip install langchain
    - pip install langchain-classic
    - pip install langchain-community
    - pip install langchain-ollama
    - pip install streamlit
    - pip install pydantic
    - pip install requests

requirements.txt -> pip freeze > requirements.txt

Ollama: https://ollama.com/
    - Açık kaynak LLM modellerini local olarak çalıştırıp kullanabilmeyi sağlayan açık kaynaklı platformdur. (eş zamanlı kullanıcı sayısı çoksa tercih edilmez..!)
    - örn. doktor asistanı projesinde Gemini'ı API ile bulut üzerinden kullandım. bu projede Ollama sayesinde açık kaynak Gemma'yı kendi bilgisayarımda on-prem olarak çalıştırabilirim.
    - llama, mistral, gemma, qwen, deepseek vb.
    - gemma3: https://ollama.com/library/gemma3
    - cmd => ollama run gemma3:4b -> ollama'ya gemma3 4b parametreli modeli kur. kurduktan sonra bu komutla cmd'den gemma3:4b çalışıtırabilirim
"""


from langchain_ollama.chat_models import ChatOllama # ollama üzerinden llm çağırmak
from langchain_classic.schema import SystemMessage, HumanMessage # sohbet mesajları sınıflandırıcısı
from langchain_classic.memory import ConversationSummaryBufferMemory # konuşma geçmişi uzadığında özetleyerek tut


llm = ChatOllama(model='gemma3:4b')

memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True)

print(f'Akıllı Turizm Rehberine hoşgeldiniz.\n'
        f'Size gezilecek yerler, tatil önerileri ve ulaşım bilgileri gibi konularda yardmcı olabilirim.')

# terminalden input alıp llm modeline gönderme ve llm modelinin cevap üretmesi
while True:
    user_input = input(f'Kullanıcı: ')
    if user_input.lower() in ["quit", "q", "bye"]:
        print(f'Program sonlandırıldı.')
        break

    memory.chat_memory.add_user_message(user_input) # kullanıcı mesajlarını hafızaya kaydet

    # model için gerekli olan mesajları oluştur. (system message, memory, human message)
    messages = [
        SystemMessage(content="Sen bir turizm rehberisin. Kullanıcılara Türkiye'deki şehirler, tarihi yerler, yöresel yemekler, ulaşım ve tatil soruları hakkında yardımcı ol.")
    ] + memory.load_memory_variables({})["history"] + [HumanMessage(content=user_input)]

    response = llm.invoke(messages) # modelden yanıt al

    memory.chat_memory.add_ai_message(response.content) # modelin cevabını hafızaya ekle

    print(f'Akıllı Turizm Rehberi: {response.content}')