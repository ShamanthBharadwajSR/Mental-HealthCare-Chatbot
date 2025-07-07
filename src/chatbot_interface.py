import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the saved embeddings and data
try:
    with open("dataset/question_embeddings.pkl", "rb") as f:
        data = pickle.load(f)  # Corrected function call
except FileNotFoundError:
    print("Error: The file 'question_embeddings.pkl' was not found. Ensure the embeddings are generated.")
    exit()

questions = data['questions']
answers = data['answers']
embeddings = np.array(data['embeddings'])

# Initialize BM25 for lexical matching
tokenized_questions = [q.lower().split() for q in questions]
bm25 = BM25Okapi(tokenized_questions)

# Load Sentence-BERT model for semantic matching
print("Loading Sentence-BERT model for semantic matching...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_hybrid_response(user_query, bm25_weight=0.5):
    """
    Hybrid retrieval: Combine BM25 and embeddings for question matching.
    Args:
        user_query (str): The user query.
        bm25_weight (float): Weight for BM25 in the hybrid score.

    Returns:
        str: The most relevant answer.
    """
    # Tokenize and compute BM25 scores
    tokenized_query = user_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Compute embeddings and cosine similarity
    query_embedding = semantic_model.encode([user_query])
    cosine_scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Combine BM25 and cosine scores
    combined_scores = (bm25_weight * bm25_scores) + ((1 - bm25_weight) * cosine_scores)

    # Get the best match
    best_index = np.argmax(combined_scores)
    return answers[best_index]

# Chatbot loop
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    response = get_hybrid_response(user_query)
    print(f"Chatbot: {response}")
