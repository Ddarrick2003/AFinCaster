# utils/sentiment.py

import tweepy
import pandas as pd
from textblob import TextBlob
import streamlit as st

# Load Twitter API token securely
BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]

# Initialize Tweepy client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(keyword, max_results=20):
    query = f"{keyword} lang:en -is:retweet"
    try:
        response = client.search_recent_tweets(query=query, tweet_fields=["created_at", "text"], max_results=max_results)
        tweets = [tweet.text for tweet in response.data] if response.data else []
        return tweets
    except Exception as e:
        st.error(f"Twitter API error: {e}")
        return []

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

def fetch_twitter_sentiment(keyword):
    tweets = fetch_tweets(keyword)
    sentiments = [analyze_sentiment(tweet) for tweet in tweets]
    df = pd.DataFrame({"Tweet": tweets, "Sentiment": sentiments})
    return df
# utils/sentiment.py

import tweepy
import pandas as pd
from textblob import TextBlob
import streamlit as st

# Load Twitter API token securely
BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]

# Initialize Tweepy client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(keyword, max_results=20):
    query = f"{keyword} lang:en -is:retweet"
    try:
        response = client.search_recent_tweets(query=query, tweet_fields=["created_at", "text"], max_results=max_results)
        tweets = [tweet.text for tweet in response.data] if response.data else []
        return tweets
    except Exception as e:
        st.error(f"Twitter API error: {e}")
        return []

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

def fetch_twitter_sentiment(keyword):
    tweets = fetch_tweets(keyword)
    sentiments = [analyze_sentiment(tweet) for tweet in tweets]
    df = pd.DataFrame({"Tweet": tweets, "Sentiment": sentiments})
    return df
