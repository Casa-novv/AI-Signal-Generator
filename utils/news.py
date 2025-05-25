# utils/news.py
import requests

def fetch_news(symbol):
    # This is a simplified example. In a real scenario, you would use a news API like NewsAPI.org
    api_key = "YOUR_NEWS_API_KEY"
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"
    response = requests.get(url)
    news = response.json()
    return news