

"""
fast api ile /ask endpointi oluşturma
uvicorn fast_api:app --reload
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.utilities import SerpAPIWrapper
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from tools.calculator_tool import calculator
from tools.custom_discount_tool import discount_calculator
from tools.rag_tool import create_rag_tool
from tools.currency_converter_tool import currency_converter


load_dotenv(r"C:\Projects\google_codes\ai_agents_yapay_zeka_asistani\.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

google_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
rag_tool = create_rag_tool(pdf_path="data/musteri_destek_faq.pdf", llm=google_llm)

tools = [
    Tool(name="search_tool", func=search_tool.run, description="google araması veya web araması yapar."),
    calculator,
    discount_calculator,
    rag_tool,
    currency_converter
]

agent = initialize_agent(tools=tools, llm=google_llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, max_iterations=5, early_stopping_method="generate", verbose=True)

# memory kullanımı
user_memories = {}
def get_user_memory(user_id:str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationSummaryBufferMemory(llm=google_llm, memory_key="chat_history", return_messages=True) # eğer user_id user_memories içinde yoksa ekler.
    return user_memories[user_id]

# fast api yapılandırması
app = FastAPI(title="Llama3.3:70b-versatile & Gemini 2.5 flash Multi-Tool AI Agent API", version="1.0")

class UserRequest(BaseModel):
    user_id: str
    message: str

@app.post("/ask")
async def ask_agent(request: UserRequest):
    user_id = request.user_id
    user_input = request.message
    memory = get_user_memory(user_id) # kullanıcı memory'sini getirme
    chat_history = "\n".join([f"Kullanıcı: {msg.content}" if msg.type == "human" else f"Agent: {msg.content}" for msg in memory.chat_memory.messages]) # bütün geçmiş konuşmayı string haline getirir ve mesaj insandan geliyorsa kullanıcı etiketini, ai'dan geliyorsa agent etiketini koyar.
    prompt_with_memory = f"{chat_history} \nkullanıcı: {user_input} \nagent:" # memory ve kullanıcı mesajını birleştirip agent'a bağlam olarak gönderilir.
    try:
        response = agent.invoke({"input": prompt_with_memory})
        ai_message = response.get("output", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(ai_message)
    return {"user_id": user_id, "response": ai_message}


"""
Greeting: Merhaba, sen kimsin
Calculator tool: bana 15 * 10 + 33 sorusunun cevabını verebilir misin?
Discount tool: telefon satmak istiyorum. telefonun fiyatı 100 tl buna indirim uygula.
Web search tool: bugün alanyada hava kaç derece?
Rag tool: merhaba ben bir ürün aldım geri iade edebilir miyim?
Memory: şimdiye kadar seninle ne konuştuk?
Exchange tool: elimde 5900 USD var. bunu TRY çevirebilir misin?
"""