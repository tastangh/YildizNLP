import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# CUDA kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model isimleri
model_names = [
    "jinaai/jina-embeddings-v3",
    "sentence-transformers/all-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1",
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

    model = model_data[1]  
    model.eval()

    question_reps = get_representation(model_data, questions)
    answer_reps = get_representation(model_data, answers)
    similarities = cosine_similarity (answer_reps,question_reps)

    top1_success = 0
    top5_success = 0

    for i in range(len(answers)):
        top_indices = np.argsort(similarities[i])[-5:]  # En yüksek benzerlikteki 5 soruyu al
        if top_indices[-1] == i:  # En benzer sorunun indeksi
            top1_success += 1
        if i in top_indices:  # Top-5 içinde mi?
            top5_success += 1

    total_answers = len(answers)

    top1_percentage = (top1_success / total_answers) * 100
    top5_percentage = (top5_success / total_answers) * 100

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

    # t-SNE uygulama ve görselleştirme
    for model_name in model_names:
        model_data = load_model(model_name)
        all_representations = np.vstack([get_representation(model_data, answers)] + 
                                        [get_representation(model_data, [question]) for question in questions])
        
        # t-SNE uygulama
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_representations)

        plt.figure(figsize=(10, 6))
        
        plt.scatter(tsne_results[:len(answers), 0], tsne_results[:len(answers), 1], 
                    label='Cevap', color='black', alpha=0.6)  # Cevap için siyah
        plt.scatter(tsne_results[len(answers):, 0], tsne_results[len(answers):, 1], 
                    label='Soru', color='red', alpha=0.6)  # Soru için kırmızı
        
        plt.title(f"{model_name} - t-SNE Görselleştirme")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid()
        
        # Grafiği PNG olarak kaydet
        plt.savefig(f"{model_name.replace('/', '_')}_tsne_visualization.png")
        plt.close()  # Grafiği kapat