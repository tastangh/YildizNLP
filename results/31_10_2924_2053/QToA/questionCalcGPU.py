import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import os
import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # To show progress bar
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import multiprocessing as mp

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

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

# Apply cleaning
questions_df['question'] = questions_df['question'].apply(clean_text)
questions_df['answer'] = questions_df['answer'].apply(clean_text)

# Simple synonym replacement for data augmentation
synonyms = {
    'iyi': 'güzel',
    'kötü': 'fena',
    'hızlı': 'çabuk',
    'yavaş': 'ağır',
    'güçlü': 'kuvvetli'
}

def augment_text(text):
    words = text.split()
    for i, word in enumerate(words):
        if word in synonyms:
            words[i] = synonyms[word] if random.random() > 0.5 else word
    return ' '.join(words)

questions_df['question'] = questions_df['question'].apply(augment_text)
questions_df['answer'] = questions_df['answer'].apply(augment_text)

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

# Hyperparameters
num_epochs = 5  # Set the number of epochs
learning_rate = 1e-5  # Set the learning rate
batch_size = 1000  # Set the mini-batch size
# patience = 2  # For early stopping

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

# Function to visualize embeddings with t-SNE
def visualize_tsne(question_embeddings, answer_embeddings, model_name):
    try:
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
    except Exception as e:
        logging.error(f"Error during TSNE visualization for {model_name}: {e}")

# Function to process a model
def process_model(model_name):
    global train_questions, train_answers
    try:
        logging.info(f"Processing model: {model_name}")
        
        # Initialize model and optimizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        patience_counter = 0

        # Store results for logging
        model_results = {
            "model_name": model_name,
            "val_top1": [],
            "val_top5": [],
            "test_top1": [],
            "test_top5": []
        }
 
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch {epoch + 1}/{num_epochs} for {model_name}...")
            model.train()  # Set the model to training mode
            
            # Shuffle training data for each epoch
            indices = np.random.permutation(len(train_questions))
            train_questions = [train_questions[i] for i in indices]
            train_answers = [train_answers[i] for i in indices]

            for i in range(0, len(train_questions), batch_size):
                batch_questions = train_questions[i:i + batch_size]
                batch_answers = train_answers[i:i + batch_size]

                # Get embeddings
                question_embeddings = get_embeddings(batch_questions, model_name)
                answer_embeddings = get_embeddings(batch_answers, model_name)

                # Calculate cosine similarity
                similarities = cosine_similarity(question_embeddings, answer_embeddings)
                angles = np.arccos(np.clip(similarities, -1, 1))

                # Find indices of top 5 answers
                top_5_indices = np.argsort(angles, axis=1)[:, :5]
                top_1_indices = np.argmin(angles, axis=1)

                # Calculate accuracy metrics
                answers_np = np.array(batch_answers)

                top_1_correct = np.sum(answers_np[top_1_indices] == answers_np[:len(top_1_indices)])
                top_5_correct = np.sum([1 if answers_np[i] in answers_np[top_5_indices[i]] else 0 for i in range(len(top_5_indices))])

                # Log training accuracy results
                train_top_1_accuracy = (top_1_correct / len(batch_questions)) * 100
                train_top_5_accuracy = (top_5_correct / len(batch_questions)) * 100

                logging.info(f"{model_name} - Epoch {epoch + 1}, Batch {i//batch_size + 1}: Training Top 1 Accuracy: {train_top_1_accuracy:.2f}%")
                logging.info(f"{model_name} - Epoch {epoch + 1}, Batch {i//batch_size + 1}: Training Top 5 Accuracy: {train_top_5_accuracy:.2f}%")

            # Validation after each epoch
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                valid_question_embeddings = get_embeddings(valid_questions, model_name)
                valid_answer_embeddings = get_embeddings(valid_answers, model_name)

                valid_similarities = cosine_similarity(valid_question_embeddings, valid_answer_embeddings)
                valid_angles = np.arccos(np.clip(valid_similarities, -1, 1))

                valid_top_5_indices = np.argsort(valid_angles, axis=1)[:, :5]
                valid_top_1_indices = np.argmin(valid_angles, axis=1)

                valid_top_1_correct = np.sum(np.array(valid_answers)[valid_top_1_indices] == np.array(valid_answers[:len(valid_top_1_indices)]))
                valid_top_5_correct = np.sum([1 if valid_answers[i] in np.array(valid_answers)[valid_top_5_indices[i]] else 0 for i in range(len(valid_top_5_indices))])

                val_top_1_accuracy = (valid_top_1_correct / len(valid_answers)) * 100
                val_top_5_accuracy = (valid_top_5_correct / len(valid_answers)) * 100

                model_results["val_top1"].append(val_top_1_accuracy)
                model_results["val_top5"].append(val_top_5_accuracy)

                logging.info(f"{model_name} - Validation Top 1 Accuracy: {val_top_1_accuracy:.2f}%")
                logging.info(f"{model_name} - Validation Top 5 Accuracy: {val_top_5_accuracy:.2f}%")

                # Check for early stopping
                if val_top_1_accuracy < best_val_loss:
                    best_val_loss = val_top_1_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # if patience_counter >= patience:
                #     logging.info(f"Early stopping triggered for {model_name} at epoch {epoch + 1}.")
                #     break
        
        # Test phase
        with torch.no_grad():
            test_question_embeddings = get_embeddings(test_questions, model_name)
            test_answer_embeddings = get_embeddings(test_answers, model_name)

            test_similarities = cosine_similarity(test_question_embeddings, test_answer_embeddings)
            test_angles = np.arccos(np.clip(test_similarities, -1, 1))

            test_top_5_indices = np.argsort(test_angles, axis=1)[:, :5]
            test_top_1_indices = np.argmin(test_angles, axis=1)

            test_top_1_correct = np.sum(np.array(test_answers)[test_top_1_indices] == np.array(test_answers[:len(test_top_1_indices)]))
            test_top_5_correct = np.sum([1 if test_answers[i] in np.array(test_answers)[test_top_5_indices[i]] else 0 for i in range(len(test_top_5_indices))])

            test_top_1_accuracy = (test_top_1_correct / len(test_answers)) * 100
            test_top_5_accuracy = (test_top_5_correct / len(test_answers)) * 100
            
            model_results["test_top1"].append(test_top_1_accuracy)
            model_results["test_top5"].append(test_top_5_accuracy)

            logging.info(f"{model_name} - Test Top 1 Accuracy: {test_top_1_accuracy:.2f}%")
            logging.info(f"{model_name} - Test Top 5 Accuracy: {test_top_5_accuracy:.2f}%")
        
            # Write accuracy results to file
            with open(accuracy_log_path, 'a') as f:
                f.write(f"{model_name}\n")
                f.write(f"Validation Top 1: {val_top_1_accuracy:.2f}%\n")
                f.write(f"Validation Top 5: {val_top_5_accuracy:.2f}%\n")
                f.write(f"Test Top 1: {test_top_1_accuracy:.2f}%\n")
                f.write(f"Test Top 5: {test_top_5_accuracy:.2f}%\n")
                f.write("\n")
    
        visualize_tsne(test_question_embeddings, test_answer_embeddings, model_name)

    except Exception as e:
        logging.error(f"Error processing model {model_name}: {e}")

if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Start multiprocessing
    with mp.Pool(processes=len(model_names)) as pool:
        pool.map(process_model, model_names)

    logging.info("Processing completed for all models.") 
