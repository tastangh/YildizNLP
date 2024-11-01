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
import torch.multiprocessing as mp

# Set up logging
log_dir = 'results/log'
os.makedirs(log_dir, exist_ok=True)

# Create a unique filename
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(log_dir, f'outputQToA_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Model names
model_names = [
    # "jinaai/jina-embeddings-v3",
    # "sentence-transformers/all-MiniLM-L12-v2",
    # "intfloat/multilingual-e5-large-instruct",
    # "BAAI/bge-m3",
    # "nomic-ai/nomic-embed-text-v1",
    "dbmdz/bert-base-turkish-cased"
]

# Load CSV file
question_answer_path = 'data/results/sampled_question_answer_for_question.csv'
try:
    logging.info("Reading CSV file...")
    questions_df = pd.read_csv(question_answer_path, sep=';')
    logging.info("CSV file successfully read.")
except Exception as e:
    logging.error(f"Error reading CSV file: {e}")
    raise

# Extract questions and answers
questions = questions_df['question'].tolist()
answers = questions_df['answer'].tolist()

# Set device
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
except Exception as e:
    logging.error(f"Error setting device: {e}")
    raise

# Function to get embeddings for a single model
def get_embeddings(texts, model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    except Exception as e:
        logging.error(f"Error loading model or tokenizer for {model_name}: {e}")
        raise
    
    embeddings = []
    try:
        logging.info(f"Getting embeddings for {model_name}...")
        for text in tqdm(texts, desc=f"Processing {model_name}", leave=False):
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Convert to float32 if necessary
            embedding = outputs.last_hidden_state.mean(dim=1).float()
            
            # Resize embeddings to a common size
            target_dim = 512  # Or the desired fixed dimension
            if embedding.size(1) > target_dim:
                embedding = embedding[:, :target_dim]
            elif embedding.size(1) < target_dim:
                padding = torch.zeros((1, target_dim - embedding.size(1))).to(device)
                embedding = torch.cat([embedding, padding], dim=1)
            
            embeddings.append(embedding)
        
        embeddings = torch.vstack(embeddings).cpu()  # Move to CPU for memory optimization
        logging.info(f"Embeddings for {model_name} obtained.")
    except Exception as e:
        logging.error(f"Error obtaining embeddings for {model_name}: {e}")
        raise

    return embeddings

# Function to process a model
def process_model(model_name):
    try:
        logging.info(f"Processing model: {model_name}")
        
        # Get embeddings
        question_embeddings = get_embeddings(questions, model_name)
        answer_embeddings = get_embeddings(answers, model_name)

        # Calculate cosine similarity
        similarities = cosine_similarity(question_embeddings, answer_embeddings)
        angles = np.arccos(np.clip(similarities, -1, 1))

        # Find indices of top 5 answers
        top_5_indices = np.argsort(angles, axis=1)[:, :5]
        
        # Top 1 accuracy (minimum angular distance)
        top_1_indices = np.argmin(angles, axis=1)

        # Determine true answers
        true_answers = answers

        # Calculate accuracy metrics
        top_1_correct = np.sum(np.array(true_answers)[top_1_indices] == true_answers)
        top_5_correct = np.sum([1 if true_answers[i] in np.array(true_answers)[top_5_indices[i]] else 0 for i in range(len(top_5_indices))])
        
        top_1_accuracy = (top_1_correct / len(questions_df)) * 100
        top_5_accuracy = (top_5_correct / len(questions_df)) * 100

        # Log accuracy results
        logging.info(f"{model_name} - Top 1 Accuracy: {top_1_accuracy:.2f}%")
        logging.info(f"{model_name} - Top 5 Accuracy: {top_5_accuracy:.2f}%")

        # # Visualization with TSNE
        # logging.info(f"Applying TSNE for {model_name}...")
        # tsne = TSNE(n_components=2, random_state=42)
        # all_embeddings = torch.cat((question_embeddings, answer_embeddings), dim=0)
        # tsne_results = tsne.fit_transform(all_embeddings)

        # # Visualization
        # plt.figure(figsize=(10, 8))
        # plt.scatter(tsne_results[:len(questions_df), 0], tsne_results[:len(questions_df), 1], label='Questions', color='blue', alpha=0.5)
        # plt.scatter(tsne_results[len(questions_df):, 0], tsne_results[len(questions_df):, 1], label='Answers', color='red', alpha=0.5)
        # plt.title(f'TSNE Visualization for {model_name}')
        # plt.xlabel('TSNE Component 1')
        # plt.ylabel('TSNE Component 2')
        # plt.legend()
        # plt.show()
        # logging.info(f"TSNE visualization completed for {model_name}.")
    
    except Exception as e:
        logging.error(f"Error processing model {model_name}: {e}")
        raise

# Initialize multiprocessing for parallel model processing
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')  # Use 'spawn' for CUDA multiprocessing compatibility
        with mp.Pool(processes=len(model_names)) as pool:
            pool.map(process_model, model_names)
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")
        raise
