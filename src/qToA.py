import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# CUDA kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model isimleri
model_names = [
    # "jinaai/jina-embeddings-v3",
    # "sentence-transformers/all-MiniLM-L12-v2",
    # "intfloat/multilingual-e5-large-instruct",
    # "BAAI/bge-m3",
    # "nomic-ai/nomic-embed-text-v1",
    "dbmdz/bert-base-turkish-cased"
]
# Modelleri ve tokenizasyonu yükleyin
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    return tokenizer, model

# CSV dosyasından veri yükleme
question_answer_path = 'data/results/sampled_question_answer.csv'
data = pd.read_csv(question_answer_path, sep=';', header=0)
data.columns = ['index', 'question', 'answer']

# Soruları ve cevapları ayır
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Temsil verilerini çıkartma
def get_representation(model_data, texts):
    tokenizer, model = model_data
    representations = []
    for text in tqdm(texts, desc="Tokenizing texts", leave=False):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state.mean(dim=1).float()
        representations.append(last_hidden_state.cpu().numpy())
    return np.vstack(representations)

# Başarıları değerlendirme
def evaluate_model(model_name):
    model_data = load_model(model_name)  # Model ve tokenizer birlikte yükleniyor

    question_reps = get_representation(model_data, questions)
    answer_reps = get_representation(model_data, answers)
    similarities = cosine_similarity(question_reps, answer_reps)

    top1_success = 0
    top5_success = 0

    for i in range(len(questions)):
        top_indices = np.argsort(similarities[i])[-5:]
        if top_indices[-1] == i:
            top1_success += 1
        if i in top_indices:
            top5_success += 1

    total_questions = len(questions)

    top1_percentage = (top1_success / total_questions) * 100
    top5_percentage = (top5_success / total_questions) * 100

    return model_name, top1_percentage, top5_percentage

# Ana işlem
if __name__ == '__main__':
    results = []
    for model_name in tqdm(model_names, desc="Evaluating models"):
        result = evaluate_model(model_name)
        results.append(result)

    with open("model_success_results.txt", "w") as f:
        f.write("Model Adı | Top 1 Başarı (%) | Top 5 Başarı (%)\n")
        f.write("-----------------------------------------\n")
        for model_name, top1, top5 in results:
            f.write(f"{model_name} | {top1:.2f} | {top5:.2f}\n")

    print("Sonuçlar model_success_results.txt dosyasına yazıldı.")
