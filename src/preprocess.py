# preprocess.py
import pandas as pd                    #loading and saving datasets
import nltk                           #Natural Language Toolkit for text preprocessing stopword removal
from nltk.tokenize import word_tokenize  #for splitting into individual tokens
from nltk.corpus import stopwords        #for removing common words
from nltk.sentiment import SentimentIntensityAnalyzer           #to categorize text
import string

# Download necessary NLTK resources
nltk.download('punkt')          #for tokenization
nltk.download('stopwords')
nltk.download('vader_lexicon')      #for analyzing the sentimnet

# Load the dataset
df = pd.read_csv('dataset/final_mentalhealth.csv')

# Initialize stopwords, punctuation, and sentiment analyzer
stop_words = set(stopwords.words('english'))        
punctuation = set(string.punctuation)
sia = SentimentIntensityAnalyzer()              #to evaluate sentiment of text

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize words and convert to lowercase
    tokens = [t for t in tokens if t not in stop_words and t not in punctuation]  # Remove stopwords and punctuation
    return ' '.join(tokens)               #Combines the processed tokensto a single string

# Sentiment Analysis function
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return "positive"
    elif sentiment_score['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply preprocessing and sentiment analysis to Questions and Answers
df['Preprocessed_Questions'] = df['Questions'].apply(preprocess_text)       #stored in new column
df['Preprocessed_Answers'] = df['Answers'].apply(preprocess_text)
df['Sentiment'] = df['Questions'].apply(get_sentiment)

# Save the preprocessed data
df.to_csv('dataset/preprocessed_data.csv', index=False)
print("Preprocessing complete. Data saved to 'dataset/preprocessed_data.csv'.")
