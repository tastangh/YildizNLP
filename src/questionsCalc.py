import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import datetime
import logging

# Günlükleme ayarları
log_dir = 'results/log'
os.makedirs(log_dir, exist_ok=True)  # Klasörü oluştur

# Benzersiz dosya adı oluştur
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(log_dir, f'outputQToA_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),  # Dosyaya yaz
        logging.StreamHandler()  # Konsola yaz
    ]
)

# Kullanılacak modeller
model_names = [
    "thenlper/gte-large",
    "sentence-transformers/all-MiniLM-L12-v2",
    "jinaai/jina-embeddings-v3",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1",
    "dbmdz/bert-base-turkish-cased"
]

# CSV dosyası
question_answer_path = 'data/results/sampled_question_answer_for_question.csv'

# CSV dosyasını oku
logging.info("CSV dosyası okunuyor...")
questions_df = pd.read_csv(question_answer_path, sep=';')
logging.info("CSV dosyası başarıyla okundu.")

# Soruları ve cevapları ayırma
questions = questions_df['question'].tolist()
answers = questions_df['answer'].tolist()  # Cevaplar sorulardan alınıyor

# Modelleri ve tokenları yükle
models = {}
tokenizers = {}
for model_name in model_names:
    try:
        logging.info(f"{model_name} yükleniyor...")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        models[model_name] = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        logging.info(f"{model_name} yüklendi.")
    except Exception as e:
        logging.error(f"{model_name} yüklenirken hata oluştu: {e}")

def get_embeddings(texts, model_name):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    embeddings = []

    logging.info(f"{model_name} için temsilleri alınıyor...")
    for text in tqdm(texts, desc=f"{model_name} için temsilleri alınıyor...", leave=False):
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))  # Temsil olarak ortalama alınır

    logging.info(f"{model_name} için temsiller alındı.")
    return torch.vstack(embeddings)  # Tüm temsilleri birleştir

# Soruların ve cevapların temsillerini al
question_embeddings = {model_name: get_embeddings(questions, model_name) for model_name in model_names}
answer_embeddings = {model_name: get_embeddings(answers, model_name) for model_name in model_names}

# Benzerlik hesaplama ve en benzer 5 cevabı bulma
top_1_results = {}
top_5_results = {}

# Sorulardan cevaplara
logging.info("Benzerlik hesaplanıyor ve en benzer 5 cevap bulunuyor...")
true_answers_array = np.array(answers)  # Cevapların diziye çevrilmiş hali
for model_name in model_names:
    similarities = cosine_similarity(question_embeddings[model_name], answer_embeddings[model_name])
    
    # Her bir soru için en benzer 5 cevabı bul
    top_5_indices = np.argsort(similarities, axis=1)[:, -5:]  # En yüksek 5 benzerlik indeksini al
    top_5_results[model_name] = top_5_indices
    
    # Top 1 başarı
    top_1_indices = np.argmax(similarities, axis=1)
    top_1_results[model_name] = top_1_indices

# Başarı oranlarını hesapla
logging.info("Başarı oranları hesaplanıyor...")
top_1_accuracy = {}
top_5_accuracy = {}
for model_name in model_names:
    top_1_correct = np.sum(true_answers_array[top_1_results[model_name]] == true_answers_array)
    top_5_correct = np.sum([
        1 if true_answers_array[i] in true_answers_array[top_5_results[model_name][i]]
        else 0 for i in range(len(top_5_results[model_name]))
    ])
    
    top_1_accuracy[model_name] = top_1_correct / len(questions_df) * 100  # Yüzde olarak başarı
    top_5_accuracy[model_name] = top_5_correct / len(questions_df) * 100  # Yüzde olarak başarı

# Başarı oranlarını yazdır
for model_name in model_names:
    logging.info(f"{model_name} - Top 1 Başarı: {top_1_accuracy[model_name]:.2f}%")
    logging.info(f"{model_name} - Top 5 Başarı: {top_5_accuracy[model_name]:.2f}%")

# Görselleştirme
for model_name in model_names:
    logging.info(f"{model_name} için TSNE uygulanıyor...")
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat((question_embeddings[model_name], answer_embeddings[model_name]), dim=0)
    tsne_results = tsne.fit_transform(all_embeddings)

    # Görselleştirme
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:len(questions_df), 0], tsne_results[:len(questions_df), 1], label='Questions', color='blue', alpha=0.5)
    plt.scatter(tsne_results[len(questions_df):, 0], tsne_results[len(questions_df):, 1], label='Answers', color='red', alpha=0.5)
    plt.title(f'TSNE Visualization for {model_name}')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.show()
    logging.info(f"{model_name} için TSNE görselleştirmesi tamamlandı.")
