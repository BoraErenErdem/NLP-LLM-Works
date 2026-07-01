

"""
Problem Tanımı:
    - Projenini amacı Google Gemini 2.5 flash modelini kullanarak ya da Groq Llama3.3 modelini kullanarak çok araçlı (multi-tool) yapay zeka aracı geliştir.
    - Ajan, kullanıcıdan gelen mesajları anlayıp, uygun aracı seçerek (tool call) görevleri otonom şekilde yerine getirecek.
    - Ajan, langchain altyapısını kullanarak gerçek dünya senaryolarını simüle eden 5 farklı tool'a sahip olur.
        - RAG -> belgelerle konuşma ve dışardan bilgi getirimi
        - Calculator -> matematiksel işlemler
        - Discount Tool (custom func) -> indirim hesaplama
        - Web Search (serpAPI) -> internet'e bağlanıp bilgi getirme
        - Memory -> konuşma geçmişini hatırlama
    - Bu sistem önce kullanıcı mesajlarını analiz eder, uygun aracı (tool) seçer, görev sonucunu üretir ve yanıtı oluşturur.
    - Geçiş ve geçmiş konuşmalarını hatırlayarak bağlamlı bir sohbet deneyimi sağlar.

Kullanılan Teknolojiler:
    - llm -> google gemini 2.5 flash modeli (tool çağırabilir)
    - langchain -> agent altyapısı, memory yönetimi, tool entegrasyonu
    - faiss vektör veri tabanı -> hızlı benzerlik araması
    - LaBSE embedding modeli -> çok dilli metin vektörleştirme, türkçe!
    - API ve arayüz -> fastapi ve streamlit (production aşamasında streamlit tercih edilmez!)
    - Diğer:
        - web search -> serpAPI
        - dotenv -> .env'den api anahtar yönetimi
        - requests -> client isteği göndermek

Kullanılacak Tool'lar:
    - RAG tool
        - belge faiss database'e çevrilir, kullanıcı mesajı embedding ile vektörleştirilir, faiss database'den benzer chunk'lar geri çağırılarak llm modeline bağlam olarak sunulur
    - Discount tool
        - ürün fiyatına %x indirim uygulama
    - Calculator
        - matematiksel işlemleri gerçekleştirme (python eval)
    - Web Search tool
        - serpAPIWrapper ile internet'ten bilgi getirimi
    - Memory
        - ConversationSummaryBufferMemory ile konuşma geçmişini hatırlamak

Plan/Program (Akış Şeması):
    - .env'den api anahtarlarını oluştur ve oku
    - tool'ları hazırla, her tool için ayrı dosyalar tanımla
    - agent oluştur, tool'ları agent'a yani llm'e bağla
    - hafıza yönetimi ile kullanıcıya özel memory nesnesi oluştur
    - her mesaj sonunda memory'i güncelle
    - /ask endpoint'i üzerinden json mesajı alınır, llm yanıtı döndürülür ve hafıza güncellenir
    - istemci katmanı (client.py) ile requests modülünü kullanarak /ask endpoint'ine istek atarak test et
    - streamlit ile arayüz oluştur

Sistem Çalışma Akışı (ÖZET):
    - kullanıcı streamlit üzerinden sorgu yapar -> fastapi /ask url -> agent ile tool seçimi (RAG / Calculator / Discount / Search / Memory) -> langchain ile tool çağrısı (call) + llm reasoning -> gemini 2.5 flash ile cevap üretimi -> memory ile geçmişi saklama -> yanıt oluştur, fastapi ile ilet ve streamlit ile kullanıcıya göster

***Sonuç olarak bu proje üretken yapay zeka ajanlarının nasıl bir şekilde düşünebilen sistemler haline geldiğini gösteren projedir. Langchain ve Gemini entegrasyonu sayesinde multi-tool, memory, rag altyapılı, etkileşimli fastapi ve streamlit arayüz destekli, mlops dev ops (docker-deployment) ile bir akıllı sistem ortaya çıkarılmış olur.***

Kütüphaneler:
    - pip install langchain
    - pip install langchain-community
    - pip install langchain-classic
    - pip install langchain-ollama
    - pip install langchain-google-genai
    - pip install google-generativeai
    - pip install faiss-cpu
    - pip install python-dotenv
    - pip install serpapi
    - pip install streamlit
    - pip install fastapi
    - pip install google-search-results
    - pip install pypdf
    - pip install sentence-transformers
    - pip install uvicorn
    - pip install requests
"""


from langchain_classic.utilities import SerpAPIWrapper # web search
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI # gemini 2.5 flash
from langchain_groq import ChatGroq # Llama 3.3:70b
from langchain_classic.agents import initialize_agent, AgentType # agent tanımlama
from langchain_classic.tools import Tool # tool tanımlama
from dotenv import load_dotenv
import os
# custom tools
from tools.calculator_tool import calculator
from tools.custom_discount_tool import discount_calculator
from tools.rag_tool import create_rag_tool
from tools.currency_converter_tool import currency_converter


load_dotenv(r"C:\Projects\google_codes\ai_agents_yapay_zeka_asistani\.env")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# tool calling yapabilen llm modelleri
google_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY) # tool 1 -> SerpAPI (genel web araması)

memory = ConversationSummaryBufferMemory(llm=google_llm, memory_key="chat_history", return_messages=True) # tool 2 -> Memory

# tool 3 -> matematiksel hesaplama
# tool 4 -> custom discount calculator
# tool 6 -> döviz birimi çevirici

rag_tool = create_rag_tool(pdf_path="data/musteri_destek_faq.pdf", llm=google_llm) # tool 5 -> RAG (vector_db'den bilgi getirme)

# tüm tool'ları agent'a ekleme
tools = [
    Tool(name="SearchTool", func=search.run, description="web araması yapar."), # sadece search için manuel Tool() ile sarmaladım çünkü bir Tool() nesnesi değil. Eğer önceden Tool() ile sarmalamış olsaydım o zaman buna gerek kalmazdı..! # func=search.run -> parantez olmadan kullandım çünkü agent fonksiyonu gerektiğinde çağırsın diye referans olarak verdim. run() olsaydı hemen çağırırdı ve hata verirdi..! # SerpAPIWrapper eski utilities sınıfı olduğundan invoke yerine run kullandım..!
    calculator, # @tool dekoratörü ile tanımlı olduğu için Tool() ile sarmalamadım.
    discount_calculator, # @tool dekoratörü ile tanımlı olduğu için Tool() ile sarmalamadım.
    rag_tool, # Tool() sarmalandığı için tekrar sarmalamadım.
    currency_converter # Tool() sarmalandığı için tekrar sarmalamadım.
]

# agent oluşturma
agent = initialize_agent(tools=tools, llm=google_llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, max_iterations=6, early_stopping_method="generate") # early_stopping_method="generate" -> limit dolunca cevap üret loop'a girme # max_iterations= -> max iterasyon sayısı
# tools= -> tool'lar  # llm= -> beyin  # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION -> tool description'larına bakarak önceki örneklere ihtiyaç duymadan hangi tool'u kullanacağına karar verir ve ReAct döngüsüyle çalışır..! # handle_parsing_errors=True -> parsing hatası varsa kendi kendine çözmeye çalışır.


# prototip kullanım testi
if __name__ == "__main__":
    print(f"Agent hazır. Konuşmaya başlayabilirsiniz.")
    while True:
        user_input = input("kullanıcı: ")
        if user_input.lower() in ["q", "quit", "exit", "bye"]:
            print(f"Agent kapatılıyor....")
            break
        chat_history = "\n".join([f"kullanıcı: {msg.content}" if msg.type == "human" else f"Agent: {msg.content}" for msg in memory.chat_memory.messages]) # mesaj insandan geliyorsa kullanıcı etiketi, agent'tan geliyorsa agent etiketi koyar.
        prompt_with_memory = f"{chat_history}\nKullanıcı: {user_input}\nAgent:"
        response = agent.invoke({"input": prompt_with_memory}) # invoke() fonksiyonu dict dönürür. bu yüzden bilgiyi dict içinden alırım.
        ai_message = response.get("output", "") # invoke() dict döndürdüğü için dict'ten string almak gerekir.

        # kullanıcı mesajı ve agent yanıtı memory'e kaydet
        memory.chat_memory.add_user_message(user_input) # kullanıcı mesajı
        memory.chat_memory.add_ai_message(ai_message) # ai cevabı
        print(f"Agent: {ai_message}\n")


"""
Greeting: Merhaba, sen kimsin
Calculator tool: bana 15 * 10 + 33 sorusunun cevabını verebilir misin?
Discount tool: telefon satmk istiyorum. telefonun fiyatı 100 tl buna indirim uygula.
Web search tool: bugün alanyada hava kaç derece?
Rag tool: merhaba ben bir ürün aldım geri iade edebilir miyim?
Memory: şimdiye kadar seninle ne konuştuk?
Exchange tool: elimde 5900 USD var. bunu TRY çevirebilir misin?
"""