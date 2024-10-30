import logging
import os
import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity  # Ensure this import is added
from tqdm import tqdm  # Import tqdm for progress bars

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

# Hyperparameters
learning_rate = 5e-5
batch_size = 8
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for DataLoader
class QuestionAnswerDataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        encoding = self.tokenizer(
            question,
            answer,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        item = {key: val.flatten() for key, val in encoding.items()}

        # Dummy label - replace with actual labels if available
        item['labels'] = torch.tensor(1)  # Ensure actual labels are provided if available

        return item

# Function to get embeddings
def get_embeddings(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting embeddings", unit='text'):
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
            output = model(**encoding)
            # Adjust this line to ensure output is 2D
            embeddings.append(output.logits.cpu().numpy().squeeze())

    return np.array(embeddings)

# Function to train the model
def train_model(model, train_loader, device):
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Use torch's AdamW
    
    for epoch in range(num_epochs):
        total_loss = 0
        # Wrap train_loader with tqdm for progress tracking
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Function to process a model
def process_model(model_name, questions, answers):
    try:
        logging.info(f"Processing model: {model_name}")

        # Create dataset and dataloader
        dataset = QuestionAnswerDataset(questions, answers)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

        # Train the model
        train_model(model, train_loader, device)

        # Get embeddings for questions and answers
        question_embeddings = get_embeddings(questions, model_name)
        answer_embeddings = get_embeddings(answers, model_name)

        # Calculate cosine similarity
        similarities = cosine_similarity(question_embeddings, answer_embeddings)
        angles = np.arccos(np.clip(similarities, -1, 1))

        # Find indices of top 5 answers
        top_5_indices = np.argsort(angles, axis=1)[:, :5]

        # Top 1 accuracy
        top_1_indices = np.argmin(angles, axis=1)

        # Placeholder true_answers list, replace with actual answers if available
        true_answers = answers  # Adjust this as per your data structure

        # Calculate accuracy metrics
        top_1_correct = np.sum(np.array(true_answers)[top_1_indices] == true_answers)
        top_5_correct = np.sum([1 if true_answers[i] in np.array(true_answers)[top_5_indices[i]] else 0 for i in range(len(top_5_indices))])

        top_1_accuracy = (top_1_correct / len(questions)) * 100
        top_5_accuracy = (top_5_correct / len(questions)) * 100

        # Log accuracy results
        logging.info(f"{model_name} - Top 1 Accuracy: {top_1_accuracy:.2f}%")
        logging.info(f"{model_name} - Top 5 Accuracy: {top_5_accuracy:.2f}%")

        # Visualization with TSNE
        logging.info(f"Applying TSNE for {model_name}...")
        tsne = TSNE(n_components=2, random_state=42)
        all_embeddings = np.concatenate((question_embeddings, answer_embeddings), axis=0)
        tsne_results = tsne.fit_transform(all_embeddings)

        # Visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:len(questions), 0], tsne_results[:len(questions), 1], label='Questions', color='blue', alpha=0.5)
        plt.scatter(tsne_results[len(questions):, 0], tsne_results[len(questions):, 1], label='Answers', color='red', alpha=0.5)
        plt.title(f'TSNE Visualization for {model_name}')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.legend()
        plt.show()
        logging.info(f"TSNE visualization completed for {model_name}.")
    
    except Exception as e:
        logging.error(f"Error processing model {model_name}: {e}")

# Main execution
if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Specify the model name
    for model_name in model_names:
        process_model(model_name, questions, answers)
