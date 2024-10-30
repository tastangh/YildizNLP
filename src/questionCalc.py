import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging
import datetime

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
    "sentence-transformers/all-MiniLM-L12-v2",
    "jinaai/jina-embeddings-v3",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
    "thenlper/gte-large",
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
answers = questions_df['answer'].tolist()

# Modelleri ve tokenları yükle
models, tokenizers = {}, {}
for model_name in model_names:
    try:
        logging.info(f"{model_name} yükleniyor...")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        models[model_name] = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        logging.info(f"{model_name} yüklendi.")
    except Exception as e:
        logging.error(f"{model_name} yüklenirken hata oluştu: {e}")

# Temsilleri çıkarma ve önbellek kontrolü
def get_embeddings(texts, model_name):
    cache_file = f'embeddings_{model_name.replace("/", "_")}.pt'
    
    # Önbellekten yükle
    if os.path.exists(cache_file):
        logging.info(f"{model_name} için temsiller önbellekten yükleniyor...")
        return torch.load(cache_file)
    
    tokenizer, model = tokenizers[model_name], models[model_name]
    embeddings = []

    logging.info(f"{model_name} için temsilleri alınıyor...")
    for text in tqdm(texts, desc=f"{model_name} için temsilleri alınıyor...", leave=False):
        # max_length 512 olarak ayarla
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))

    embeddings_tensor = torch.vstack(embeddings)  # Tüm temsilleri birleştir

    # Temsilleri önbelleğe kaydet
    torch.save(embeddings_tensor, cache_file)
    logging.info(f"{model_name} için temsiller alındı ve önbelleğe kaydedildi.")
    return embeddings_tensor

# Soruların ve cevapların temsillerini al
question_embeddings = {model_name: get_embeddings(questions, model_name) for model_name in model_names}
answer_embeddings = {model_name: get_embeddings(answers, model_name) for model_name in model_names}

# Açı benzerliği hesaplama fonksiyonu
def calculate_angle_similarity(embeddings_1, embeddings_2):
    similarities = []
    for i in range(len(embeddings_1)):
        A = embeddings_1[i].numpy()
        B = embeddings_2.numpy()
        cosine_sim = cosine_similarity([A], B)
        angle = np.arccos(np.clip(cosine_sim, -1.0, 1.0))  # Açı hesaplama
        similarities.append(angle)
    return np.array(similarities)

# Top-1 ve Top-5 başarıları hesaplama fonksiyonu
def calculate_top_accuracy_with_angle(embeddings_1, embeddings_2, true_labels, df_len):
    angle_similarities = calculate_angle_similarity(embeddings_1, embeddings_2)
    top_5_indices = np.argsort(angle_similarities, axis=1)[:, :5]  # Küçük açılar
    top_1_indices = np.argmin(angle_similarities, axis=1)  # En küçük açı
    top_1_correct = np.sum(np.array(true_labels)[top_1_indices] == true_labels)
    top_5_correct = np.sum([1 if true_labels[i] in np.array(true_labels)[top_5_indices[i]] else 0 for i in range(df_len)])
    return top_1_correct / df_len * 100, top_5_correct / df_len * 100

# Başarıları hesaplama
top_1_accuracy, top_5_accuracy = {}, {}
for model_name in model_names:
    # Sorulardan cevaplara benzerlik hesaplama
    top_1_acc_q2a, top_5_acc_q2a = calculate_top_accuracy_with_angle(
        question_embeddings[model_name], answer_embeddings[model_name], answers, len(questions_df)
    )
    
    top_1_accuracy[model_name] = top_1_acc_q2a
    top_5_accuracy[model_name] = top_5_acc_q2a
    logging.info(f"{model_name} - Top 1 Başarı: {top_1_accuracy[model_name]:.2f}%, Top 5 Başarı: {top_5_accuracy[model_name]:.2f}%")

# Görselleştirme
for model_name in model_names:
    # TSNE uygulama
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