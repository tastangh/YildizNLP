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

def calculate_loss(angles):
    """
    Calculate the loss based on the angles between question and answer embeddings.
    
    Args:
        angles (numpy.ndarray): Array of angles between question and answer embeddings.
    
    Returns:
        torch.Tensor: Calculated loss value.
    """
    # Create a tensor from angles with requires_grad set to True
    angles_tensor = torch.tensor(angles, dtype=torch.float32, requires_grad=True).to(device)

    # Loss calculation (mean of angles)
    loss = torch.mean(angles_tensor)
    
    return loss


def train_and_validate_model(model_name, questions_train, answers_train, questions_val, answers_val, num_epochs=2, batch_size=800, learning_rate=1e-5):
    """
    Train the specified model with the given questions and answers, and validate after each epoch.
    
    Args:
        model_name (str): Name of the model to train.
        questions_train (list): List of questions for training.
        answers_train (list): List of corresponding answers.
        questions_val (list): List of questions for validation.
        answers_val (list): List of corresponding answers.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Size of the training batches.
        learning_rate (float): Learning rate for the optimizer.
    """
    logging.info(f"Training model: {model_name}")

    # Initialize the optimizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training loop
        model.train()
        for i in tqdm(range(0, len(questions_train), batch_size), desc=f"Training {model_name}", leave=False):
            # Prepare batch
            batch_questions = questions_train[i:i + batch_size]
            batch_answers = answers_train[i:i + batch_size]

            # Get embeddings for the batch
            question_embeddings_train = get_embeddings(batch_questions, model_name)
            answer_embeddings_train = get_embeddings(batch_answers, model_name)

            # Calculate cosine similarity for training
            similarities_train = cosine_similarity(question_embeddings_train, answer_embeddings_train)
            angles_train = np.arccos(np.clip(similarities_train, -1, 1))

            # Define a loss function
            loss = calculate_loss(angles_train)  # Implement your own loss function
            optimizer.zero_grad()  # Zero gradients before backward pass
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Calculate accuracy metrics for training (Top-1 and Top-5 accuracy)
            top_1_indices_train = np.argmin(angles_train, axis=1)
            top_5_indices_train = np.argsort(angles_train, axis=1)[:, :5]  # Get Top-5 indices
            top_1_correct_train = np.sum(np.array(batch_answers)[top_1_indices_train] == batch_answers)
            top_5_correct_train = np.sum(np.isin(batch_answers, np.array(batch_answers)[top_5_indices_train]))  # Check if answers are in Top-5
            top_1_accuracy_train = (top_1_correct_train / len(batch_questions)) * 100
            top_5_accuracy_train = (top_5_correct_train / len(batch_questions)) * 100

            logging.info(f"Training Batch {i // batch_size + 1}: Top 1 Accuracy: {top_1_accuracy_train:.2f}%, Top 5 Accuracy: {top_5_accuracy_train:.2f}%")
        
        # Validation step (similar to training but without backpropagation)
        model.eval()
        logging.info("Validating model...")
        question_embeddings_val = get_embeddings(questions_val, model_name)
        answer_embeddings_val = get_embeddings(answers_val, model_name)

        # Calculate cosine similarity for validation
        similarities_val = cosine_similarity(question_embeddings_val, answer_embeddings_val)
        angles_val = np.arccos(np.clip(similarities_val, -1, 1))

        # Calculate accuracy metrics for validation (Top-1 and Top-5 accuracy)
        top_1_indices_val = np.argmin(angles_val, axis=1)
        top_5_indices_val = np.argsort(angles_val, axis=1)[:, :5]  # Get Top-5 indices
        top_1_correct_val = np.sum(np.array(answers_val)[top_1_indices_val] == answers_val)
        top_5_correct_val = np.sum(np.isin(answers_val, np.array(answers_val)[top_5_indices_val]))  # Check if answers are in Top-5
        top_1_accuracy_val = (top_1_correct_val / len(questions_val)) * 100
        top_5_accuracy_val = (top_5_correct_val / len(questions_val)) * 100

        logging.info(f"Validation: Top 1 Accuracy: {top_1_accuracy_val:.2f}%, Top 5 Accuracy: {top_5_accuracy_val:.2f}%")

def test_model(model_name, questions_test, answers_test):
    """
    Test the specified model with the given questions and answers.
    
    Args:
        model_name (str): Name of the model to test.
        questions_test (list): List of questions for testing.
        answers_test (list): List of corresponding answers.
    """
    logging.info("Testing model...")
    question_embeddings_test = get_embeddings(questions_test, model_name)
    answer_embeddings_test = get_embeddings(answers_test, model_name)

    # Calculate cosine similarity for test
    similarities_test = cosine_similarity(question_embeddings_test, answer_embeddings_test)
    angles_test = np.arccos(np.clip(similarities_test, -1, 1))

    # Calculate accuracy metrics for test
    top_1_indices_test = np.argmin(angles_test, axis=1)
    top_5_indices_test = np.argsort(angles_test, axis=1)[:, :5]  # Get Top-5 indices
    top_1_correct_test = np.sum(np.array(answers_test)[top_1_indices_test] == answers_test)
    top_5_correct_test = np.sum(np.isin(answers_test, np.array(answers_test)[top_5_indices_test]))  # Check if answers are in Top-5
    top_1_accuracy_test = (top_1_correct_test / len(questions_test)) * 100
    top_5_accuracy_test = (top_5_correct_test / len(questions_test)) * 100

    logging.info(f"Test Results: Top 1 Accuracy: {top_1_accuracy_test:.2f}%, Top 5 Accuracy: {top_5_accuracy_test:.2f}%")

def run_models():
    for model_name in model_names:
        # Train and validate the model
        train_and_validate_model(model_name, questions_train, answers_train, questions_val, answers_val)

        # Test the model
        test_model(model_name, questions_test, answers_test)

if __name__ == "__main__":
    run_models()
