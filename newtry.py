import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Model Selection and Loading
model_names = [
    'intfloat/multilingual-e5-large-instruct',
    'HIT-TMG/KaLM-embedding-multilingual-mini-v1',
    'Alibaba-NLP/gte-multilingual-base',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'dbmdz/bert-base-turkish-cased'  # Reference model
]

# Load models and tokenizers
logger.info("Loading models and tokenizers...")
models = {}
tokenizers = {}

for name in model_names:
    models[name] = AutoModel.from_pretrained(name, trust_remote_code=True)
    tokenizers[name] = AutoTokenizer.from_pretrained(name)

logger.info("Models and tokenizers loaded successfully.")

# 2. Data Preparation
logger.info("Preparing data...")
df = pd.read_csv("question_answer.csv", encoding='utf-8-sig', sep=';')

# Check if the required columns exist in the DataFrame
if 'question' not in df.columns or 'answer' not in df.columns:
    logger.error("The DataFrame must contain 'question' and 'answer' columns.")
    raise ValueError("The DataFrame must contain 'question' and 'answer' columns.")

questions = df['question'].sample(1000).tolist()
answers = df['answer'].sample(1000).tolist()
logger.info(f"Data prepared: {len(questions)} questions and {len(answers)} answers.")

# 3. Generate Embeddings
def get_embeddings(texts, model_name):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average of the last layer
    return embeddings

logger.info("Generating embeddings for questions and answers...")
question_embeddings = {}
answer_embeddings = {}

for name in tqdm(model_names, desc="Generating question embeddings"):
    question_embeddings[name] = get_embeddings(questions, name)

for name in tqdm(model_names, desc="Generating answer embeddings"):
    answer_embeddings[name] = get_embeddings(answers, name)

logger.info("Embeddings generated successfully.")

# 4. Similarity Measurement
def calculate_similarity(embeddings_a, embeddings_b):
    similarity = F.cosine_similarity(embeddings_a.unsqueeze(1), embeddings_b.unsqueeze(0), dim=-1)
    return similarity

logger.info("Calculating similarity matrices...")
similarity_matrices = {}
for model_name in tqdm(model_names, desc="Calculating similarity"):
    similarity_matrices[model_name] = calculate_similarity(question_embeddings[model_name], answer_embeddings[model_name])

logger.info("Similarity matrices calculated successfully.")

# 5. Calculate Accuracy Metrics
def calculate_top_k_accuracy(similarity_matrix, true_indices, k=5):
    top_k_indices = torch.topk(similarity_matrix, k, dim=-1).indices
    top1_accuracy = (top_k_indices[:, 0] == true_indices).float().mean().item()
    top5_accuracy = (top_k_indices == true_indices.unsqueeze(1)).float().max(dim=-1)[0].mean().item()
    return top1_accuracy, top5_accuracy

# Define true indices (assumed to be aligned with the sampled data)
true_question_indices = torch.tensor(range(1000))  # Adjust based on your data
true_answer_indices = torch.tensor(range(1000))  # Adjust based on your data

logger.info("Calculating accuracy metrics...")
success_rates = {}
for model_name in tqdm(model_names, desc="Calculating accuracy"):
    success_rates[model_name] = {
        'top1': calculate_top_k_accuracy(similarity_matrices[model_name], true_question_indices, k=1),
        'top5': calculate_top_k_accuracy(similarity_matrices[model_name], true_answer_indices, k=5)
    }

logger.info("Accuracy metrics calculated successfully.")

# 6. t-SNE Visualization
def plot_tsne(embeddings, model_name):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings.numpy())
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', label='Questions', alpha=0.5)
    plt.title(f't-SNE Visualization for {model_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

logger.info("Generating t-SNE visualizations...")
for model_name in tqdm(model_names, desc="Visualizing embeddings"):
    plot_tsne(question_embeddings[model_name], model_name)
    plot_tsne(answer_embeddings[model_name], model_name)

# Print Results
logger.info("Printing results...")
for model_name, rates in success_rates.items():
    print(f"Model: {model_name}, Top-1 Accuracy: {rates['top1']:.4f}, Top-5 Accuracy: {rates['top5']:.4f}")

logger.info("Process completed successfully.")
