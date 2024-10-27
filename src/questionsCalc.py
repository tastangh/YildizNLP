import pandas as pd
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm  # İlerleme çubuğu için

# Hugging Face model isimleri
models = [
    "intfloat/multilingual-e5-large-instruct",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "Alibaba-NLP/gte-multilingual-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "dbmdz/bert-base-turkish-cased"
]

# Cache dizini belirleme
cache_dir = "cache/models/"
os.makedirs(cache_dir, exist_ok=True)

# Sampled questions ve sampled answers dosyalarını oku
questions_path = 'data/results/sampled_questions.csv'
sampled_questions = pd.read_csv(questions_path, sep=';')

# Model temsilleri almak için fonksiyon
def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] token'ının çıktısını al
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Başarı hesaplama
def calculate_top_k_accuracy(true_answer, predicted_answers, k=5):
    # Gerçek cevabın içinde olup olmadığını kontrol et
    return int(true_answer in predicted_answers[:k])

# Modelleri yükleyip temsilleri hesapla
results = []

# Tüm modeller için döngü
for model_name in tqdm(models, desc="Model Yükleme ve Hesaplama", position=0):
    # Model ve tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    # Sorular ve cevaplar için döngü
    model_results = {
        'top1': 0,
        'top5': 0,
        'total': len(sampled_questions)
    }

    # Tüm soruların temsillerini hesapla
    for index, row in tqdm(sampled_questions.iterrows(), total=sampled_questions.shape[0], desc=f"{model_name} - Soru İşleme", position=1):
        question = row['question']
        answer = row['answer']
        answer_candidates = sampled_questions['answer'].tolist()  # Tüm cevaplar
        
        # Soru ve cevap için temsilleri al
        question_embedding = get_embedding(model, tokenizer, question)
        answer_embedding = get_embedding(model, tokenizer, answer)

        # Diğer tüm cevaplar ile benzerlik hesapla
        answer_embeddings = [get_embedding(model, tokenizer, candidate) for candidate in tqdm(answer_candidates, desc="Cevap Temsilleri Hesaplama", leave=False)]
        similarities = cosine_similarity(question_embedding.reshape(1, -1), answer_embeddings).flatten()

        # En benzer 5 cevabı bul
        top_indices = np.argsort(similarities)[-5:][::-1]  # En yüksek benzerlik için indeksleri al
        top_answers = [answer_candidates[i] for i in top_indices]

        # Başarı hesapla
        model_results['top1'] += calculate_top_k_accuracy(answer, top_answers, k=1)
        model_results['top5'] += calculate_top_k_accuracy(answer, top_answers, k=5)

    results.append({
        'model': model_name,
        'top1_accuracy': model_results['top1'] / model_results['total'],
        'top5_accuracy': model_results['top5'] / model_results['total']
    })

# Sonuçları yazdır
for result in results:
    print(f"{result['model']} - Top-1 Doğruluğu: {result['top1_accuracy']:.4f}, Top-5 Doğruluğu: {result['top5_accuracy']:.4f}")
