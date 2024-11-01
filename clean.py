import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
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
accuracy_log_path = os.path.join(log_dir, f'accuracy_results_{timestamp}.txt')

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

# Split the data into train (80%), validation (10%), and test (10%) sets
train_questions, temp_questions, train_answers, temp_answers = train_test_split(
    questions, answers, test_size=0.2, random_state=42)
valid_questions, test_questions, valid_answers, test_answers = train_test_split(
    temp_questions, temp_answers, test_size=0.5, random_state=42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")



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
            
            embedding = outputs.last_hidden_state.mean(dim=1).float()
            embeddings.append(embedding)
        
        embeddings = torch.vstack(embeddings).cpu()
        logging.info(f"Embeddings for {model_name} obtained.")
    except Exception as e:
        logging.error(f"Error obtaining embeddings for {model_name}: {e}")
        raise

    return embeddings


if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Start multiprocessing
    with mp.Pool(processes=len(model_names)) as pool:
        pool.map(process_model, model_names)

    logging.info("Processing completed for all models.") 
