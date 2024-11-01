import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
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
        embedding = outputs.last_hidden_state.mean(dim=1).float()
        embeddings.append(embedding)
    
    embeddings = torch.vstack(embeddings).cpu()
    logging.info(f"Embeddings for {model_name} obtained.")
    return embeddings

def visualize_tsne(model_name, question_embeddings, answer_embeddings):
    logging.info(f"Applying TSNE for {model_name}...")
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat((question_embeddings, answer_embeddings), dim=0)
    tsne_results = tsne.fit_transform(all_embeddings.cpu().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:len(question_embeddings), 0], tsne_results[:len(question_embeddings), 1], label='Questions', color='blue', alpha=0.5)
    plt.scatter(tsne_results[len(question_embeddings):, 0], tsne_results[len(question_embeddings):, 1], label='Answers', color='red', alpha=0.5)
    plt.title(f'TSNE Visualization for {model_name}')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.show()
    logging.info(f"TSNE visualization completed for {model_name}.")

def train_and_validate_model(model_name):
    logging.info(f"Training model: {model_name}")

    # Initialize the optimizer and model outside the training loop
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=10e-5)

    for epoch in range(1):  # Change number of epochs as needed
        logging.info(f"Epoch {epoch + 1}/1")
        
        model.train()
        question_embeddings_train = get_embeddings(questions_train, model_name)
        answer_embeddings_train = get_embeddings(answers_train, model_name)

        # Calculate cosine similarity for training
        similarities_train = cosine_similarity(question_embeddings_train, answer_embeddings_train)
        angles_train = np.arccos(np.clip(similarities_train, -1, 1))

        # Calculate accuracy metrics for training (Top-1 and Top-5 accuracy)
        top_1_indices_train = np.argmin(angles_train, axis=1)
        top_1_correct_train = np.sum(np.array(answers_train)[top_1_indices_train] == np.array(answers_train))
        top_1_accuracy_train = (top_1_correct_train / len(questions_train)) * 100

        # Calculate Top-5 Accuracy
        top_5_indices_train = np.argsort(angles_train, axis=1)[:, :5]  # Get the indices of the top 5
        top_5_correct_train = np.sum(np.array(answers_train)[top_5_indices_train] == np.array(answers_train)[:, np.newaxis])
        top_5_accuracy_train = (top_5_correct_train / len(questions_train)) * 100

        logging.info(f"Training: Top 1 Accuracy: {top_1_accuracy_train:.2f}%")
        logging.info(f"Training: Top 5 Accuracy: {top_5_accuracy_train:.2f}%")

        # Validation
        model.eval()
        question_embeddings_val = get_embeddings(questions_val, model_name)
        answer_embeddings_val = get_embeddings(answers_val, model_name)

        similarities_val = cosine_similarity(question_embeddings_val, answer_embeddings_val)
        angles_val = np.arccos(np.clip(similarities_val, -1, 1))

        top_1_indices_val = np.argmin(angles_val, axis=1)
        top_1_correct_val = np.sum(np.array(answers_val)[top_1_indices_val] == np.array(answers_val))
        top_1_accuracy_val = (top_1_correct_val / len(questions_val)) * 100

        # Calculate Top-5 Accuracy for validation
        top_5_indices_val = np.argsort(angles_val, axis=1)[:, :5]  # Get the indices of the top 5
        top_5_correct_val = np.sum(np.array(answers_val)[top_5_indices_val] == np.array(answers_val)[:, np.newaxis])
        top_5_accuracy_val = (top_5_correct_val / len(questions_val)) * 100

        logging.info(f"Validation: Top 1 Accuracy: {top_1_accuracy_val:.2f}%")
        logging.info(f"Validation: Top 5 Accuracy: {top_5_accuracy_val:.2f}%")
        visualize_tsne(model_name, question_embeddings_val, answer_embeddings_val)

def test_model(model_name):
    logging.info("Testing model...")
    question_embeddings_test = get_embeddings(questions_test, model_name)
    answer_embeddings_test = get_embeddings(answers_test, model_name)

    similarities_test = cosine_similarity(question_embeddings_test, answer_embeddings_test)
    angles_test = np.arccos(np.clip(similarities_test, -1, 1))

    top_1_indices_test = np.argmin(angles_test, axis=1)
    top_1_correct_test = np.sum(np.array(answers_test)[top_1_indices_test] == np.array(answers_test))
    top_1_accuracy_test = (top_1_correct_test / len(questions_test)) * 100

    # Calculate Top-5 Accuracy for testing
    top_5_indices_test = np.argsort(angles_test, axis=1)[:, :5]  # Get the indices of the top 5
    top_5_correct_test = np.sum(np.array(answers_test)[top_5_indices_test] == np.array(answers_test)[:, np.newaxis])
    top_5_accuracy_test = (top_5_correct_test / len(questions_test)) * 100

    logging.info(f"Test Results: Top 1 Accuracy: {top_1_accuracy_test:.2f}%")
    logging.info(f"Test Results: Top 5 Accuracy: {top_5_accuracy_test:.2f}%")
    visualize_tsne(model_name, question_embeddings_test, answer_embeddings_test)

def run_models(model_name):
    train_and_validate_model(model_name)
    test_model(model_name)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')  # Necessary for CUDA with multiprocessing
        with mp.Pool(processes=len(model_names)) as pool:
            pool.map(run_models, model_names)
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")
