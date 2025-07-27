# utils/sentiment.py

import snscrape.modules.twitter as sntwitter
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# ---------- Twitter Sentiment using VADER ----------
def fetch_twitter_sentiment(query, max_tweets=100):
    analyzer = SentimentIntensityAnalyzer()
    tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        tweets.append(tweet.content)

    df = pd.DataFrame(tweets, columns=['Tweet'])
    df['Sentiment'] = df['Tweet'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
    df['Label'] = df['Sentiment'].apply(lambda s: 'Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral')
    
    return df

# ---------- News Sentiment using FinBERT ----------
def fetch_news_sentiment(news_list):
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
    results = classifier(news_list)
    
    df = pd.DataFrame(news_list, columns=['Headline'])
    df['Label'] = [r['label'] for r in results]
    df['Score'] = [r['score'] for r in results]
    
    return df
