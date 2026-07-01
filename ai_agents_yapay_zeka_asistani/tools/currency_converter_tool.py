

"""
para birimi dönüştürücü
"""

from langchain_classic.tools import Tool
import requests

def get_exchange_money(currency: str):
    """
    döviz çevirme işlemi yapar. örn: "100 usd to try" ya da "50 eur to usd" vb.
    """
    try:
        parts = currency.upper().split()
        amount = float(parts[0])
        from_currency = parts[1]
        to_currency = parts[3]
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        response = requests.get(url)
        data = response.json()
        if to_currency not in data["rates"]:
            return f"error: {to_currency} döviz kuru bulunamadı."
        rate = data["rates"][to_currency]
        result = amount * rate
        return f"answer: {amount} {from_currency} = {result:.2f} {to_currency}"
    except Exception as e:
        return f"error: {e}"

currency_converter = Tool(name="currency_converter", func=get_exchange_money, description="Güncel kurla döviz çevirme işlemi yapar. Kullanıcı 'X dolar tl' veya 'X euro dolar' gibi doğal dilde yazsa bile input formatını '100 USD to TRY' şeklinde dönüştürerek kullan. örnek: '500 USD to TRY', '50 EUR to USD'")

if __name__ == "__main__":
    print(get_exchange_money("100 USD to TRY"))