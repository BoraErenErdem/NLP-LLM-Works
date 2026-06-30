

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