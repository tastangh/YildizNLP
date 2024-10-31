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
        train_question_embeddings = get_embeddings(train_questions, model_name)
        train_answer_embeddings = get_embeddings(train_answers, model_name)
        valid_question_embeddings = get_embeddings(valid_questions, model_name)
        valid_answer_embeddings = get_embeddings(valid_answers, model_name)
        test_question_embeddings = get_embeddings(test_questions, model_name)
        test_answer_embeddings = get_embeddings(test_answers, model_name)

        # Calculate cosine similarity for training
        train_similarities = cosine_similarity(train_question_embeddings, train_answer_embeddings)
        train_angles = np.arccos(np.clip(train_similarities, -1, 1))

        # Find indices of top 5 answers for training
        train_top_5_indices = np.argsort(train_angles, axis=1)[:, :5]
        train_top_1_indices = np.argmin(train_angles, axis=1)

        # Calculate accuracy metrics for training
        train_top_1_correct = np.sum(np.array(train_answers)[train_top_1_indices] == train_answers)
        train_top_5_correct = np.sum([1 if train_answers[i] in np.array(train_answers)[train_top_5_indices[i]] else 0 for i in range(len(train_top_5_indices))])
        
        train_top_1_accuracy = (train_top_1_correct / len(train_questions)) * 100
        train_top_5_accuracy = (train_top_5_correct / len(train_questions)) * 100

        # Log training accuracy results
        logging.info(f"{model_name} - Training Top 1 Accuracy: {train_top_1_accuracy:.2f}%")
        logging.info(f"{model_name} - Training Top 5 Accuracy: {train_top_5_accuracy:.2f}%")

        # Validation similarity calculations
        valid_similarities = cosine_similarity(valid_question_embeddings, valid_answer_embeddings)
        valid_angles = np.arccos(np.clip(valid_similarities, -1, 1))

        # Find indices of top 5 answers for validation
        valid_top_5_indices = np.argsort(valid_angles, axis=1)[:, :5]
        valid_top_1_indices = np.argmin(valid_angles, axis=1)

        # Calculate accuracy metrics for validation
        valid_top_1_correct = np.sum(np.array(valid_answers)[valid_top_1_indices] == valid_answers)
        valid_top_5_correct = np.sum([1 if valid_answers[i] in np.array(valid_answers)[valid_top_5_indices[i]] else 0 for i in range(len(valid_top_5_indices))])
        
        valid_top_1_accuracy = (valid_top_1_correct / len(valid_questions)) * 100
        valid_top_5_accuracy = (valid_top_5_correct / len(valid_questions)) * 100

        # Log validation accuracy results
        logging.info(f"{model_name} - Validation Top 1 Accuracy: {valid_top_1_accuracy:.2f}%")
        logging.info(f"{model_name} - Validation Top 5 Accuracy: {valid_top_5_accuracy:.2f}%")

        # Test similarity calculations
        test_similarities = cosine_similarity(test_question_embeddings, test_answer_embeddings)
        test_angles = np.arccos(np.clip(test_similarities, -1, 1))

        # Find indices of top 5 answers for test
        test_top_5_indices = np.argsort(test_angles, axis=1)[:, :5]
        test_top_1_indices = np.argmin(test_angles, axis=1)

        # Calculate accuracy metrics for test
        test_top_1_correct = np.sum(np.array(test_answers)[test_top_1_indices] == test_answers)
        test_top_5_correct = np.sum([1 if test_answers[i] in np.array(test_answers)[test_top_5_indices[i]] else 0 for i in range(len(test_top_5_indices))])
        
        test_top_1_accuracy = (test_top_1_correct / len(test_questions)) * 100
        test_top_5_accuracy = (test_top_5_correct / len(test_questions)) * 100

        # Log test accuracy results
        logging.info(f"{model_name} - Test Top 1 Accuracy: {test_top_1_accuracy:.2f}%")
        logging.info(f"{model_name} - Test Top 5 Accuracy: {test_top_5_accuracy:.2f}%")

        # Visualization with TSNE
        logging.info(f"Applying TSNE for {model_name}...")
        tsne = TSNE(n_components=2, random_state=42)
        all_embeddings = torch.cat((train_question_embeddings, train_answer_embeddings, valid_question_embeddings, valid_answer_embeddings), dim=0)
        tsne_results = tsne.fit_transform(all_embeddings)

        # Visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:len(train_questions), 0], tsne_results[:len(train_questions), 1], label='Train Questions', color='blue', alpha=0.5)
        plt.scatter(tsne_results[len(train_questions):len(train_questions)+len(train_answers), 0], tsne_results[len(train_questions):len(train_questions)+len(train_answers), 1], label='Train Answers', color='red', alpha=0.5)
        plt.scatter(tsne_results[len(train_questions)+len(train_answers):, 0], tsne_results[len(train_questions)+len(train_answers):, 1], label='Valid Questions', color='green', alpha=0.5)
        plt.scatter(tsne_results[len(train_questions)+len(train_answers):, 0], tsne_results[len(train_questions)+len(train_answers):, 1], label='Valid Answers', color='orange', alpha=0.5)
        plt.title(f'TSNE Visualization for {model_name}')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.legend()
        # plt.show()
        logging.info(f"TSNE visualization completed for {model_name}.")
        
    except Exception as e:
        logging.error(f"Error processing model {model_name}: {e}")
        raise

# Main processing function
def main():
    for model_name in model_names:
        process_model(model_name)

if __name__ == "__main__":
    main()  