# =========================
# 📁 FILE: utils/sentiment.py
# =========================

import snscrape.modules.twitter as sntwitter
import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# -------------------------
# 📘 VADER Twitter Sentiment
# -------------------------
def get_twitter_sentiment(query="stock market", limit=50):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query + ' since:2023-01-01').get_items()):
        if i >= limit:
            break
        tweets.append(tweet.content)

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in tweets:
        score = analyzer.polarity_scores(text)
        sentiments.append({"Tweet": text, "Compound": score["compound"]})

    df = pd.DataFrame(sentiments)
    df['Sentiment'] = df['Compound'].apply(lambda c: 'Positive' if c > 0.05 else 'Negative' if c < -0.05 else 'Neutral')
    return df

# -------------------------
# 📰 FinBERT News Sentiment
# -------------------------

def get_news_headlines(query="stock market", max_articles=10):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')

    articles = []
    for article in soup.select('article h3')[:max_articles]:
        title = article.text.strip()
        articles.append(title)
    return articles

def get_finbert_sentiment(headlines):
    classifier = pipeline("text-classification", model="ProsusAI/finbert")
    results = classifier(headlines)

    df = pd.DataFrame({
        "Headline": headlines,
        "Sentiment": [r['label'] for r in results],
        "Score": [r['score'] for r in results]
    })
    return df

# -------------------------
# 📊 Summary Chart Data
# -------------------------

def summarize_sentiments(df, source="twitter"):
    if source == "twitter":
        summary = df['Sentiment'].value_counts(normalize=True).to_dict()
    else:
        summary = df['Sentiment'].value_counts(normalize=True).to_dict()
    return summary
