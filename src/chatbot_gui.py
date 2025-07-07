import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the saved embeddings and data
try:
    with open("dataset/question_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
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

# GUI Functions
def send_message(event=None):
    user_message = user_input.get()
    if not user_message.strip():
        return

    chat_display.configure(state=tk.NORMAL)

    # Add user message on the right
    add_message("You", user_message, "user")
    response = get_hybrid_response(user_message)

    # Add bot response on the left
    add_message("Bot", response, "bot")
    user_input.delete(0, tk.END)

    chat_display.configure(state=tk.DISABLED)

def add_message(sender, message, sender_type):
    if sender_type == "user":
        tag = "user_message"
        alignment = "right"
        bubble_bg = "#d1f5d3"
    else:
        tag = "bot_message"
        alignment = "left"
        bubble_bg = "#e8e8e8"

    chat_display.tag_configure(tag, justify=alignment, background=bubble_bg, wrap=tk.WORD, spacing3=10)

    # Add sender name
    chat_display.insert(tk.END, f"{sender}:\n", tag)
    # Add the message text
    chat_display.insert(tk.END, f"{message}\n\n", tag)
    chat_display.yview(tk.END)

# Create GUI
root = tk.Tk()
root.title("Emotion-Aware Chatbot")
root.geometry("600x700")

# Style configuration
style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background="#f4f7fc")
style.configure("TButton", background="#4caf50", foreground="#fff", font=("Arial", 14), padding=10)

# Gradient background
background_frame = tk.Frame(root, bg="#f4f7fc")
background_frame.pack(fill=tk.BOTH, expand=True)

# Header
header = tk.Label(background_frame, text="Chat with Mental Health Bot 🤖", font=("Arial", 20, "bold"), bg="#4a90e2", fg="#fff")
header.pack(fill=tk.X, pady=10)

# Chat display area
chat_display = scrolledtext.ScrolledText(background_frame, wrap=tk.WORD, font=("Arial", 12), bg="#f8f9fa", fg="#333")
chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_display.configure(state=tk.DISABLED)

# Input field
user_input = tk.Entry(background_frame, font=("Arial", 14), bg="#fff", fg="#333", relief=tk.GROOVE)
user_input.pack(padx=10, pady=5, fill=tk.X)
user_input.bind("<Return>", send_message)

# Send button
send_button = ttk.Button(background_frame, text="Send", command=send_message)
send_button.pack(padx=10, pady=5)

# Footer
footer = tk.Label(background_frame, text="Powered by Hybrid Retrieval AI", font=("Arial", 10), bg="#f4f7fc", fg="#888")
footer.pack(pady=5)

# Run GUI
root.mainloop()
