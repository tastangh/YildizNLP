import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the CSV file
df = pd.read_csv("/home/dev/workspace/YildizNLP/question_answer.csv", encoding='utf-8-sig', sep=';')

# Data preprocessing
df['question'] = df['question'].str.lower().str.strip()
df['answer'] = df['answer'].str.lower().str.strip()

# Load the models and tokenizers
model_names = [
    'distilbert-base-multilingual-cased',
    'bert-base-multilingual-cased',
    'xlm-roberta-base',
    'mt5-small',
    't5-small',
    'dbmdz/bert-base-turkish-cased'
]

tokenizers = {name: BertTokenizer.from_pretrained(name) for name in model_names}
models = {name: BertModel.from_pretrained(name).eval() for name in model_names}

def encode_texts_in_batches(texts, model, tokenizer, batch_size=32):
    """Vectorize texts in batches using the specified model."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch.tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(embeddings)

# Prepare a dictionary to store embeddings for each model
embeddings = {}

# Vectorize questions and answers using each model
for model_name in model_names:
    X_question = encode_texts_in_batches(df['question'], models[model_name], tokenizers[model_name])
    X_answer = encode_texts_in_batches(df['answer'], models[model_name], tokenizers[model_name])
    embeddings[model_name] = (X_question, X_answer)

def calculate_top_n_success(data, target_col, model_embeddings, n=5, random_samples=1000):
    # Select random samples
    sampled_indices = random.sample(range(len(data)), random_samples)
    
    top1_count = 0
    top5_count = 0
    
    # Use tqdm to show progress for the loop
    for idx in tqdm(sampled_indices, desc=f"Evaluating {target_col}"):
        target = data.iloc[idx][target_col]
        X, _ = model_embeddings

        # Compute the similarity matrix
        similarities = cosine_similarity(X[idx].reshape(1, -1), X).flatten()

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

# Evaluate for each model
for model_name, (X_question, X_answer) in embeddings.items():
    print(f"\nModel: {model_name}")
    
    # Evaluate questions based on answers
    top1_question, top5_question = calculate_top_n_success(df, 'answer', (X_question, X_answer))
    print(f"Top-1 success rate for questions based on answers: {top1_question:.2f}")
    print(f"Top-5 success rate for questions based on answers: {top5_question:.2f}")

    # Evaluate answers based on questions
    top1_answer, top5_answer = calculate_top_n_success(df, 'question', (X_answer, X_question))
    print(f"Top-1 success rate for answers based on questions: {top1_answer:.2f}")
    print(f"Top-5 success rate for answers based on questions: {top5_answer:.2f}")

# t-SNE Visualization
def visualize_embeddings(model_name, embeddings):
    X_question, X_answer = embeddings
    all_embeddings = np.vstack((X_question, X_answer))
    labels = ['Question'] * len(X_question) + ['Answer'] * len(X_answer)

    tsne = TSNE(n_components=2, random_state=42)
    embedded_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 6))
    for label in set(labels):
        plt.scatter(embedded_2d[np.array(labels) == label, 0], 
                    embedded_2d[np.array(labels) == label, 1], 
                    label=label, alpha=0.5)
    plt.title(f't-SNE Visualization of {model_name} Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid()
    plt.show()

# Visualize embeddings for each model
for model_name, (X_question, X_answer) in embeddings.items():
    visualize_embeddings(model_name, (X_question, X_answer))
