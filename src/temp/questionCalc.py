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
    # "jinaai/jina-embeddings-v3",
    # "sentence-transformers/all-MiniLM-L12-v2",
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

# Cache directory
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)  # Create cache directory

# Function to save model embeddings
def save_embeddings(embeddings, model_name):
    embeddings_path = os.path.join(cache_dir, f'embeddings_{model_name}.pt')
    torch.save(embeddings, embeddings_path)  # Save embeddings

# Function to load embeddings
def load_embeddings(model_name):
    embeddings_path = os.path.join(cache_dir, f'embeddings_{model_name}.pt')
    
    # Check if embeddings exist
    if os.path.exists(embeddings_path):
        embeddings = torch.load(embeddings_path)
        logging.info(f"Loaded embeddings for {model_name} from cache.")
        return embeddings
    else:
        logging.info(f"Embeddings for {model_name} not found in cache. Generating new embeddings...")
        return None

# Function to get embeddings for a single model
def get_embeddings(texts, model_name):
    embeddings = load_embeddings(model_name)
    
    if embeddings is not None:
        return embeddings  # Return cached embeddings

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    embeddings = []
    logging.info(f"Getting embeddings for {model_name}...")
    
    for text in tqdm(texts, desc=f"Processing {model_name}", leave=False):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))  # Average representation

    embeddings = torch.vstack(embeddings)  # Combine all embeddings

    # Save only if embeddings do not exist
    if not os.path.exists(os.path.join(cache_dir, f'embeddings_{model_name}.pt')):
        save_embeddings(embeddings, model_name)  # Save to cache
        logging.info(f"Embeddings for {model_name} obtained and cached.")
    
    return embeddings

# Calculate similarities and find top 1 and top 5 answers for each model
for model_name in model_names:
    logging.info(f"Processing model: {model_name}")
    
    # Get embeddings
    question_embeddings = get_embeddings(questions, model_name)
    answer_embeddings = get_embeddings(answers, model_name)

    # Calculate cosine similarity
    similarities = cosine_similarity(question_embeddings, answer_embeddings)
    angles = np.arccos(np.clip(similarities, -1, 1))  # Avoid small number errors

    # Find indices of top 5 answers
    top_5_indices = np.argsort(angles, axis=1)[:, :5]
    
    # Top 1 accuracy (minimum angular distance)
    top_1_indices = np.argmin(angles, axis=1)

    # Determine true answers
    true_answers = answers

    # Calculate accuracy metrics
    top_1_correct = np.sum(np.array(true_answers)[top_1_indices] == true_answers)
    top_5_correct = np.sum([1 if true_answers[i] in np.array(true_answers)[top_5_indices[i]] else 0 for i in range(len(top_5_indices))])
    
    top_1_accuracy = (top_1_correct / len(questions_df)) * 100  # Percentage accuracy
    top_5_accuracy = (top_5_correct / len(questions_df)) * 100  # Percentage accuracy

    # Log accuracy results
    logging.info(f"{model_name} - Top 1 Accuracy: {top_1_accuracy:.2f}%")
    logging.info(f"{model_name} - Top 5 Accuracy: {top_5_accuracy:.2f}%")

    # Visualization with TSNE
    logging.info(f"Applying TSNE for {model_name}...")
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat((question_embeddings, answer_embeddings), dim=0)
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
    logging.info(f"TSNE visualization completed for {model_name}.")
