import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

# Modellerin isimleri ve parametreleri
model_names = [
    'intfloat/multilingual-e5-large-instruct',
    'HIT-TMG/KaLM-embedding-multilingual-mini-v1',
    'Alibaba-NLP/gte-multilingual-base',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'dbmdz/bert-base-turkish-cased'  # Referans model
]

# Model ve tokenizer'ları yükle
tokenizers = {}
models = {}
for model_name in model_names:
    try:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        models[model_name] = AutoModel.from_pretrained(model_name).eval()
        print(f"{model_name} başarıyla yüklendi.")
    except Exception as e:
        print(f"{model_name} yüklenirken hata oluştu: {e}")

# CSV dosyasını yükle
df = pd.read_csv("question_answer.csv", encoding='utf-8-sig', sep=';')

# Metin ön işleme
df['question'] = df['question'].str.lower().str.strip()
df['answer'] = df['answer'].str.lower().str.strip()

def encode_texts_in_batches(texts, model, tokenizer, batch_size=32):
    """Verileri partiler halinde encode et."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch.tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(embeddings)

# Embedding'leri hesapla
embeddings = {}
for model_name in model_names:
    X_question = encode_texts_in_batches(df['question'], models[model_name], tokenizers[model_name])
    X_answer = encode_texts_in_batches(df['answer'], models[model_name], tokenizers[model_name])
    embeddings[model_name] = (X_question, X_answer)

def calculate_top_n_success(data, target_col, model_embeddings, n=5, random_samples=1000):
    """Top-1 ve Top-5 başarı oranlarını hesapla."""
    sampled_indices = random.sample(range(len(data)), random_samples)
    
    top1_count = 0
    top5_count = 0
    
    for idx in sampled_indices:
        target = data.iloc[idx][target_col]
        X, _ = model_embeddings
        
        # Benzerlik matrisini hesapla
        similarities = cosine_similarity(X[idx].reshape(1, -1), X).flatten()
        
        # En yüksek benzerlikleri bul
        similar_indices = np.argsort(similarities)[-(n+1):-1][::-1]
        
        # Top-1 ve Top-5 başarılarını kontrol et
        if target in data.iloc[similar_indices[:1]][target_col].values:
            top1_count += 1
        if target in data.iloc[similar_indices[:n]][target_col].values:
            top5_count += 1
            
    top1_success_rate = top1_count / random_samples
    top5_success_rate = top5_count / random_samples
    
    return top1_success_rate, top5_success_rate

# Her model için başarıları hesapla
for model_name, (X_question, X_answer) in embeddings.items():
    print(f"\nModel: {model_name}")
    
    # Cevaplara göre soruları değerlendir
    top1_answer, top5_answer = calculate_top_n_success(df, 'answer', (X_question, X_answer))
    print(f"Top-1 success rate for answers based on questions: {top1_answer:.2f}")
    print(f"Top-5 success rate for answers based on questions: {top5_answer:.2f}")

    # Sorulara göre cevapları değerlendir
    top1_question, top5_question = calculate_top_n_success(df, 'question', (X_answer, X_question))
    print(f"Top-1 success rate for questions based on answers: {top1_question:.2f}")
    print(f"Top-5 success rate for questions based on answers: {top5_question:.2f}")

# t-SNE uygulama ve görselleştirme
def plot_tsne(embeddings, labels, model_name):
    """t-SNE ile veriyi görselleştir."""
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette='viridis', alpha=0.7)
    plt.title(f't-SNE visualization for {model_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper right')
    plt.show()

# Her model için t-SNE ile görselleştir
for model_name, (X_question, X_answer) in embeddings.items():
    all_embeddings = np.vstack((X_question, X_answer))
    labels = ['question'] * len(X_question) + ['answer'] * len(X_answer)
    plot_tsne(all_embeddings, labels, model_name)
