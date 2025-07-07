# 🧠 Mental Health Support Chatbot 🤖

A hybrid chatbot built for mental health support using both semantic understanding (Sentence-BERT) and sentiment analysis.
It provides emotionally aware responses through a user-friendly web interface built with Flask.

---

![Screenshot 2024-12-20 113249](https://github.com/user-attachments/assets/b586d75f-9310-4025-bbdc-c7eba9117d30)

## 🧠 About the Project

This project is a conversational AI chatbot designed to support users in dealing with mental health-related queries. It uses a combination of:

- **Semantic Matching** with a fine-tuned Sentence-BERT model
- **Sentiment Analysis** using VADER to detect user emotion
- **Flask Web App** for an interactive browser-based chat interface

The chatbot intelligently retrieves answers based on both **semantic relevance** and **user sentiment**, enhancing user interaction and emotional context.

## 🛠 Technologies Used

- **Python 3.x**
- **Flask** – Web framework
- **NLTK** – Tokenization and Sentiment Analysis
- **SentenceTransformers (Sentence-BERT)** – Semantic embeddings
- **Scikit-learn** – Cosine similarity calculation

---

## 🌟 Features

- ✅ Semantic understanding using Sentence-BERT
- ✅ Fine-tuned BERT on domain-specific question-answer pairs
- ✅ Emotionally aware response generation (😊 / 😐 / 😔)
- ✅ Real-time chatting using a web interface

---

## Steps to run :
1. Create virtual environment
2. Install dependencies
3. Preprocess the data : python src/preprocess.py
4. Fine-Tune the Sentence-BERT Model : python src/fine_tune.py
5. Generate Embeddings: python src/generate_embeddings.py
6. Launch the app:
     ## Two interfaces have been provide you can use whichever is required


     python src/app.py

     python src/chatbot_gui.py
