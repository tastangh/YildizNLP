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
from sklearn.model_selection import train_test_split
import torch.optim as optim

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

# Split the dataset into training, validation, and test sets (80% train, 10% val, 10% test)
questions_train, questions_temp, answers_train, answers_temp = train_test_split(questions, answers, test_size=0.2, random_state=42)
questions_val, questions_test, answers_val, answers_test = train_test_split(questions_temp, answers_temp, test_size=0.5, random_state=42)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Function to get embeddings for a single model
def get_embeddings(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    embeddings = []
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
    return embeddings