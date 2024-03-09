import requests

def get_currency_price(currency_code):
    # Coinbase API endpoint for currency price
    url = f"https://api.coinbase.com/v2/prices/{currency_code.upper()}-USD/spot"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Extracting the current price of the given currency
        currency_price = data['data']['amount']
        
        return float(currency_price)
    
    except Exception as e:
        print("Error occurred:", e)
        return None

# Example usage:
currency_code = "ETH"  # Change this to any desired currency code
currency_price = get_currency_price(currency_code)
if currency_price is not None:
    print(f"Current price of {currency_code}: ${currency_price:.2f}")