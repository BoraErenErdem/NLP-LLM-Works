

"""
ürün fiyatını alır ve %10 indirim uygular
"""

from langchain_classic.tools import tool
import re

@tool("discount_tool")
def discount_calculator(product_info: str):
    """
    ürün fiyatını alır ve %10 indirim uygular.
    örn: "Elma fiyatı 50 tl" -> "indirimli fiyat: 45 tl"
    """
    try:
        price = float(re.findall(r"\d+", product_info)[0])
        discounted = price * 0.9
        return f"dicounted price: {discounted:.2f} tl"
    except Exception as e:
        return f"fiyat bulunamadı.\nerror: {e}"

print(discount_calculator.invoke("Elma fiyatı 100 tl."))