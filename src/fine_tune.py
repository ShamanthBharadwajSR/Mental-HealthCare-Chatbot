import pandas as pd
from torch.utils.data import DataLoader         #Handles batching and shuffling of data for efficient training.
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator  #Evaluates model performance on binary classification tasks.
import os       #for creating directories
import datasets

# Load the preprocessed data
df = pd.read_csv('dataset/preprocessed_data.csv')

# Prepare data for fine-tuning
train_examples = [
    InputExample(
        texts=[row['Preprocessed_Questions'], row['Preprocessed_Answers']],
        label=1.0
    ) for _, row in df.iterrows()
]

# Split into training and evaluation sets
train_size = int(0.8 * len(train_examples))
train_dataset = train_examples[:train_size]
eval_dataset = train_examples[train_size:]

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')     #to compute dense embeddings of text

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model)

# Create DataLoader for training
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# Define the evaluator
evaluator = BinaryClassificationEvaluator.from_input_examples(eval_dataset, name='mental_health-eval')

# Fine-tune the model
output_dir = 'models/fine_tuned_sentence_bert'
os.makedirs(output_dir, exist_ok=True)

print("Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    evaluation_steps=100,
    output_path=output_dir
)

print("Fine-tuning complete.", output_dir)
