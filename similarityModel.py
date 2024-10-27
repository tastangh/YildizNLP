import pandas as pd
import numpy as np
import random
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import os
import joblib  # Cache işlemleri için

# Etkileşimli grafikler için TkAgg backend'i kullan
matplotlib.use('TkAgg')

# Modellerin isimleri ve parametreleri
model_names = [
    'intfloat/multilingual-e5-large-instruct',
    'HIT-TMG/KaLM-embedding-multilingual-mini-v1',
    'Alibaba-NLP/gte-multilingual-base',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'dbmdz/bert-base-turkish-cased'
]

# Model ve tokenizer'ları yükle
tokenizers = {}
models = {}
for model_name in tqdm(model_names, desc="Modeller yükleniyor"):
    try:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        models[model_name] = AutoModel.from_pretrained(model_name).eval()
        print(f"{model_name} başarıyla yüklendi.")
    except Exception as e:
        print(f"{model_name} yüklenirken hata oluştu: {e}")

# CSV dosyasını yükle
print("CSV dosyası yükleniyor...")
df = pd.read_csv("question_answer.csv", encoding='utf-8-sig', sep=';')
print("CSV dosyası başarıyla yüklendi.")

# Metin ön işleme
print("Metin ön işleme işlemi başlıyor...")
df['question'] = df['question'].str.lower().str.strip()
df['answer'] = df['answer'].str.lower().str.strip()
print("Metin ön işleme tamamlandı.")

# Rastgele 1000 soru ve cevabı seç
random_questions = df.sample(n=1000, random_state=42)
random_answers = df.sample(n=1000, random_state=35)

# Cache kontrolü ve embedding'leri hesaplama
def get_embeddings(model_name, texts, model, tokenizer, cache_dir="embedding_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{model_name}_embeddings.pkl")
    
    if os.path.exists(cache_path):
        print(f"{model_name} için önceden hesaplanmış embedding'ler yükleniyor...")
        return joblib.load(cache_path)
    
    embeddings = encode_texts_in_batches(texts, model, tokenizer)
    joblib.dump(embeddings, cache_path)
    print(f"{model_name} için embedding'ler hesaplanıp kaydedildi.")
    return embeddings

# Text'i encoding yapmak için bir yardımcı fonksiyon
def encode_texts_in_batches(texts, model, tokenizer, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings oluşturuluyor"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch.tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(embeddings)

# Her model için embedding'leri yükle veya hesapla
embeddings = {}
for model_name in tqdm(model_names, desc="Modellerin embedding'leri hesaplanıyor"):
    X_question = get_embeddings(model_name, random_questions['question'], models[model_name], tokenizers[model_name])
    X_answer = get_embeddings(model_name, random_answers['answer'], models[model_name], tokenizers[model_name])
    embeddings[model_name] = (X_question, X_answer)

# Başarı oranlarını hesaplama işlevi
def calculate_top_n_success(data, target_col, model_embeddings, n=5):
    top1_count = 0
    top5_count = 0

    for idx in tqdm(range(len(data)), desc="Başarı oranları hesaplanıyor"):
        target = data.iloc[idx][target_col]
        X, _ = model_embeddings
        
        similarities = cosine_similarity(X[idx].reshape(1, -1), X).flatten()
        similar_indices = np.argsort(similarities)[-(n+1):-1][::-1]
        
        if target in data.iloc[similar_indices[:1]][target_col].values:
            top1_count += 1
        if target in data.iloc[similar_indices[:n]][target_col].values:
            top5_count += 1
            
    top1_success_rate = top1_count / len(data)
    top5_success_rate = top5_count / len(data)
    
    return top1_success_rate, top5_success_rate

# Her model için başarı oranlarını hesapla
for model_name, (X_question, X_answer) in embeddings.items():
    print(f"\nModel: {model_name}")
    
    top1_answer, top5_answer = calculate_top_n_success(random_answers, 'answer', (X_question, X_answer))
    print(f"Top-1 başarı oranı (soru bazlı): {top1_answer:.2f}")
    print(f"Top-5 başarı oranı (soru bazlı): {top5_answer:.2f}")

    top1_question, top5_question = calculate_top_n_success(random_questions, 'question', (X_answer, X_question))
    print(f"Top-1 başarı oranı (cevap bazlı): {top1_question:.2f}")
    print(f"Top-5 başarı oranı (cevap bazlı): {top5_question:.2f}")

# t-SNE ile görselleştirme işlevi
def plot_tsne(embeddings, labels, model_name):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette='viridis', alpha=0.7)
    plt.title(f't-SNE visualization for {model_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper right')
    plt.show()

# Her model için t-SNE ile görselleştirme
for model_name, (X_question, X_answer) in embeddings.items():
    all_embeddings = np.vstack((X_question, X_answer))
    labels = ['question'] * len(X_question) + ['answer'] * len(X_answer)
    plot_tsne(all_embeddings, labels, model_name)
