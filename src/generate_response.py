# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import pickle

# # Load the model and tokenizers
# model = load_model('models/chatbot_model.keras')

# with open('models/input_tokenizer.pkl', 'rb') as handle:
#     input_tokenizer = pickle.load(handle)

# with open('models/target_tokenizer.pkl', 'rb') as handle:
#     target_tokenizer = pickle.load(handle)

# # Function to generate a response with temperature sampling
# def generate_response(input_text, temperature=0.7, max_length=100):
#     input_sequence = input_tokenizer.texts_to_sequences([input_text])
#     input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=51)

#     start_token = target_tokenizer.word_index.get('<start>', None)
#     stop_token = target_tokenizer.word_index.get('<end>', None)
#     target_sequence = np.array([[start_token]]) if start_token is not None else np.array([[0]])

#     generated_text = ''

#     for _ in range(max_length):
#         prediction = model.predict([input_sequence, target_sequence], verbose=0)
#         prediction = prediction[0, -1, :]  # Focus on the last time step's prediction

#         # Apply temperature
#         prediction = np.log(prediction + 1e-7) / temperature
#         prediction = np.exp(prediction) / np.sum(np.exp(prediction))

#         predicted_id = np.random.choice(len(prediction), p=prediction)
        
#         if stop_token is not None and predicted_id == stop_token:
#             break

#         word = target_tokenizer.index_word.get(predicted_id, '')
#         if word:
#             generated_text += ' ' + word

#         target_sequence = np.zeros((1, 1))
#         target_sequence[0, 0] = predicted_id

#     return generated_text.strip()

# # Example usage
# input_text = "Hello, how are you?"
# response = generate_response(input_text, temperature=0.7)
# print("Chatbot response:", response)
# import pickle
# import numpy as np

# file_path = 'dataset\preprocessed_data.pkl'
# with open(file_path, 'rb') as file:
#     preprocessed_data = pickle.load(file)

# print(preprocessed_data.keys())  # Check available keys
# print("Input sequences sample:", preprocessed_data['input_sequences'][:2])
# print("Target sequences sample:", preprocessed_data['target_sequences'][:2])
# print("Max encoder sequence length:", preprocessed_data['max_encoder_seq_length'])
# print("Max decoder sequence length:", preprocessed_data['max_decoder_seq_length'])


# from tensorflow.keras.models import load_model

# try:
#     model = load_model('models\chatbot_model.h5')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading the model: {e}")
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(f"Layer: {layer.name}, Weights: {len(weights)}")

import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Add this import

# Load the tokenizers and preprocessed data
with open('dataset\preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

input_tokenizer = data['input_tokenizer']
target_tokenizer = data['target_tokenizer']
max_encoder_seq_length = data['max_encoder_seq_length']

# Test with a sample input
sample_input = "What is mental health?"
sample_input_seq = input_tokenizer.texts_to_sequences([sample_input.lower()])
sample_input_seq = pad_sequences(sample_input_seq, maxlen=max_encoder_seq_length, padding='post')

# Make prediction
encoder_model, decoder_model = None, None  # Recreate encoder and decoder models here
# Assuming you already have code to define `encoder_model` and `decoder_model`.
response = decode_sequence(sample_input_seq)  # Replace `decode_sequence` with your implementation
print(f"Response: {response}")

