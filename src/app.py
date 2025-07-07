# app.py
from flask import Flask, render_template, request   #used to create web application
import numpy as np
import pickle       #to load pre saved embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity      #computes cosine similarity for best match

# Load the saved embeddings and data
with open("dataset/question_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

questions = data['original_questions']
answers = data['answers']
embeddings = np.array(data['embeddings'])       #converting to array format
sentiments = data['sentiments']

# Load Sentence-BERT model for semantic matching
semantic_model = SentenceTransformer('models/fine_tuned_sentence_bert')


# Flask app initialization
app = Flask(__name__)

def get_response_with_sentiment(user_query):
    # Compute query embedding
    query_embedding = semantic_model.encode([user_query])
    
    # Compute cosine similarity and converts result to 1D array
    cosine_scores = cosine_similarity(query_embedding, embeddings).flatten()
    
    # Get the best match
    best_index = np.argmax(cosine_scores)
    best_answer = answers[best_index]
    detected_sentiment = sentiments[best_index]

    # Modify response based on sentiment
    if detected_sentiment == "positive":
        return f"😊 {best_answer}"
    elif detected_sentiment == "negative":
        return f"😔 {best_answer}"
    else:
        return f"😐 {best_answer}"

@app.route("/")
def home():
    return render_template("index.html")        #Renders it as the homepage of the app

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["user_message"]     #accessing user message
    chatbot_response = get_response_with_sentiment(user_message)
    return {"response": chatbot_response}       #Sends response as a JSON object to the frontend.

if __name__ == "__main__":
    app.run(debug=True)     #launches flask app
