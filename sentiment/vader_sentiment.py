import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import datetime
import os

# Load Twitter API credentials from environment or secure file
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

def fetch_tweets(query="stock market", count=100):
    auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    tweets = api.search_tweets(q=query, lang="en", count=count, tweet_mode='extended')
    data = [{"text": tweet.full_text, "created_at": tweet.created_at} for tweet in tweets]
    return pd.DataFrame(data)

def analyze_vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment"] = df["compound"].apply(lambda x: "positive" if x > 0.05 else "negative" if x < -0.05 else "neutral")
    return df
