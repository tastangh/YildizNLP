import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm  # Import tqdm for progress bars

# Load the CSV file
df = pd.read_csv("/home/dev/workspace/YildizNLP/question_answer.csv", encoding='utf-8-sig', sep=';')

# Data preprocessing
df['question'] = df['question'].str.lower().str.strip()
df['answer'] = df['answer'].str.lower().str.strip()

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')
model.eval()  # Set the model to evaluation mode

def encode_texts_in_batches(texts, batch_size=32):
    """Vectorize texts in batches using BERT."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts in batches"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch.tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Get word embedding vectors from BERT's output
        embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(embeddings)

# Vectorize questions and answers using BERT in batches
X_question = encode_texts_in_batches(df['question'])
X_answer = encode_texts_in_batches(df['answer'])

def calculate_top_n_success(data, target_col, n=5, random_samples=1000):
    # Select random samples
    sampled_indices = random.sample(range(len(data)), random_samples)
    
    top1_count = 0
    top5_count = 0
    
    # Use tqdm to show progress for the loop
    for idx in tqdm(sampled_indices, desc=f"Evaluating {target_col}"):
        target = data.iloc[idx][target_col]
        
        # Compute the similarity matrix
        if target_col == 'question':
            similarities = cosine_similarity(X_question[idx].reshape(1, -1), X_question).flatten()
        else:
            similarities = cosine_similarity(X_answer[idx].reshape(1, -1), X_answer).flatten()

        # Find the highest similarities
        similar_indices = np.argsort(similarities)[-(n+1):-1][::-1]  # Get Top-N, excluding own index
        
        # Remove own index if present
        if idx in similar_indices:
            similar_indices = np.delete(similar_indices, np.where(similar_indices == idx))
        
        # Check Top-1 and Top-5 successes
        if target in data.iloc[similar_indices[:1]][target_col].values:
            top1_count += 1
        if target in data.iloc[similar_indices[:n]][target_col].values:
            top5_count += 1
            
    top1_success_rate = top1_count / random_samples
    top5_success_rate = top5_count / random_samples
    
    return top1_success_rate, top5_success_rate

# Evaluate questions based on answers
top1_question, top5_question = calculate_top_n_success(df, 'answer', n=5, random_samples=1000)
print(f"Top-1 success rate for questions based on answers: {top1_question:.2f}")
print(f"Top-5 success rate for questions based on answers: {top5_question:.2f}")

# Evaluate answers based on questions
top1_answer, top5_answer = calculate_top_n_success(df, 'question', n=5, random_samples=1000)
print(f"Top-1 success rate for answers based on questions: {top1_answer:.2f}")
print(f"Top-5 success rate for answers based on questions: {top5_answer:.2f}")
