import numpy as np
import random

def generate_mock_sentiment(df):
    np.random.seed(42)
    df = df.copy()
    df['Sentiment'] = [random.uniform(-1, 1) for _ in range(len(df))]
    return df
