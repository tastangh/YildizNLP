import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging
import os
import datetime

# Set up logging
log_dir = 'results/log'
os.makedirs(log_dir, exist_ok=True)  # Create directory

# Create a unique filename
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(log_dir, f'outputQToA_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),  # Write to file
        logging.StreamHandler()  # Write to console
    ]
)

# Model names
model_names = [
    # "sentence-transformers/all-MiniLM-L12-v2",
    # "jinaai/jina-embeddings-v3",
    # "intfloat/multilingual-e5-large-instruct",
    # "BAAI/bge-m3",
    # "thenlper/gte-large",
    # "nomic-ai/nomic-embed-text-v1",
    "dbmdz/bert-base-turkish-cased"
]

# Load CSV file
question_answer_path = 'data/results/sampled_question_answer_for_question.csv'
logging.info("Reading CSV file...")
questions_df = pd.read_csv(question_answer_path, sep=';')
logging.info("CSV file successfully read.")

# Extract questions and answers
questions = questions_df['question'].tolist()
answers = questions_df['answer'].tolist()

# Load models and tokenizers
models, tokenizers = {}, {}
for model_name in model_names:
    try:
        logging.info(f"Loading model: {model_name}...")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        models[model_name] = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        logging.info(f"{model_name} loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading {model_name}: {e}")

# Function to get embeddings
def get_embeddings(texts, model_name):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    embeddings = []

    logging.info(f"Getting embeddings for {model_name}...")
    for text in tqdm(texts, desc=f"Processing {model_name}", leave=False):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))  # Average representation

    logging.info(f"Embeddings for {model_name} obtained.")
    return torch.vstack(embeddings)  # Combine all embeddings

# Get embeddings for questions and answers
question_embeddings = {model_name: get_embeddings(questions, model_name) for model_name in model_names}
answer_embeddings = {model_name: get_embeddings(answers, model_name) for model_name in model_names}

# Calculate similarities and find top 1 and top 5 answers
top_1_results = {}
top_5_results = {}

logging.info("Calculating similarities and finding top 5 answers...")
for model_name in model_names:
    # Calculate cosine similarity
    similarities = cosine_similarity(question_embeddings[model_name], answer_embeddings[model_name])
    angles = np.arccos(np.clip(similarities, -1, 1))  # Avoid small number errors

    # Find indices of top 5 answers
    top_5_indices = np.argsort(angles, axis=1)[:, :5]
    top_5_results[model_name] = top_5_indices

    # Top 1 accuracy (minimum angular distance)
    top_1_indices = np.argmin(angles, axis=1)
    top_1_results[model_name] = top_1_indices

# Determine true answers
true_answers = answers

# Calculate accuracy metrics
top_1_accuracy = {}
top_5_accuracy = {}

logging.info("Calculating accuracy metrics...")
for model_name in model_names:
    top_1_correct = np.sum(np.array(true_answers)[top_1_results[model_name]] == true_answers)
    top_5_correct = np.sum([1 if true_answers[i] in np.array(true_answers)[top_5_results[model_name][i]] else 0 for i in range(len(top_5_results[model_name]))])
    
    top_1_accuracy[model_name] = (top_1_correct / len(questions_df)) * 100  # Percentage accuracy
    top_5_accuracy[model_name] = (top_5_correct / len(questions_df)) * 100  # Percentage accuracy

# Log accuracy results
for model_name in model_names:
    logging.info(f"{model_name} - Top 1 Accuracy: {top_1_accuracy[model_name]:.2f}%")
    logging.info(f"{model_name} - Top 5 Accuracy: {top_5_accuracy[model_name]:.2f}%")


# Visualization with TSNE
for model_name in model_names:
    logging.info(f"Applying TSNE for {model_name}...")
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat((question_embeddings[model_name], answer_embeddings[model_name]), dim=0)
    tsne_results = tsne.fit_transform(all_embeddings)

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:len(questions_df), 0], tsne_results[:len(questions_df), 1], label='Questions', color='blue', alpha=0.5)
    plt.scatter(tsne_results[len(questions_df):, 0], tsne_results[len(questions_df):, 1], label='Answers', color='red', alpha=0.5)
    plt.title(f'TSNE Visualization for {model_name}')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.show()
    logging.info(f"{model_name} için TSNE görselleştirmesi tamamlandı.")