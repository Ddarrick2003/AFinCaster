from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    pred_label = labels[torch.argmax(probs)]
    confidence = probs.max().item()
    return pred_label, confidence

def analyze_news_sentiment(news_df):
    news_df["prediction"], news_df["confidence"] = zip(*news_df["text"].apply(predict_sentiment))
    return news_df
