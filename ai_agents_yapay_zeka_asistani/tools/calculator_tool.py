

"""
temel matematik hesaplayıcı
"""

from langchain_classic.tools import tool # dekoratör

@tool("calculator_tool") # @tool dekoratörü -> fonksiyonu langchain agent sistemine özel tool nesnesine çevirir. bu yüzden fonksiyon çağrılacağı zaman .invoke() ile çağrılır. bu sayede agent'lar bu tool'u otomatik olarak çağırabilir.
def calculator(expression: str):
    # !!! docstring içindeki bilgi agent'ın tool'u ne zaman kullanacağını belirtir..!
    """
    basit matematiksel ifadeleri değerlendirir.
    örnek: '25 * (5 + 3)' -> 'answer: 200'
    """
    try:
        result = eval(expression)
        return f"answer: {result}" # "answer:" kelimesi ReAct ajanlarının sonucu daha rahat anlamasını sağlar.
    except Exception as e:
        return f"error: {e}"

# print(calculator.invoke("25 * (5 + 3)"))