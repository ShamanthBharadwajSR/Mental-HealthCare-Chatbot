# generate_embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer  #generating embeddings
import numpy as np              
import pickle       #for saving embeddings

# Load the preprocessed data
df = pd.read_csv('dataset/preprocessed_data.csv')

# Load a pre-trained Sentence-BERT model
print("Loading Sentence-BERT model...")
model = SentenceTransformer('models/fine_tuned_sentence_bert')


# Compute embeddings for the preprocessed questions
print("Generating embeddings for questions...")
question_embeddings = model.encode(df['Preprocessed_Questions'].tolist(), convert_to_numpy=True)

# Save the embeddings and corresponding data
with open('dataset/question_embeddings.pkl', 'wb') as f:
    pickle.dump({
        'questions': df['Preprocessed_Questions'].tolist(),
        'original_questions': df['Questions'].tolist(),
        'answers': df['Answers'].tolist(),
        'sentiments': df['Sentiment'].tolist(),
        'embeddings': question_embeddings
    }, f)

print("Embeddings generated and saved successfully.")
