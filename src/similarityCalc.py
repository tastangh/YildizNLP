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

# Benzerlik hesaplama fonksiyonu
def compute_similarity(model, tokenizer, text1, text2):
    # Metinleri encode et
    inputs = tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()

    # Cosine similarity hesapla
    similarity = cosine_similarity(embeddings)
    
    return similarity[0][1]  # Cosine benzerliği döndür

# Sonuçları saklayacağımız liste
results = []

# Sampled questions ve sampled answers dosyalarını oku
questions_path = 'data/results/sampled_questions.csv'
answers_path = 'data/results/sampled_answers.csv'

sampled_questions = pd.read_csv(questions_path, sep=';')
sampled_answers = pd.read_csv(answers_path, sep=';')

# Modelleri yükleyip çalıştırma
for model_name in models:
    print(f"Processing with model: {model_name}")

    # Model ve tokenizer yükleme
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    # Rastgele 1000 soru ve cevap ile benzerlik hesaplama
    similarities = []
    for index in tqdm(range(1000), desc="Processing"):
        question = sampled_questions.iloc[index]['question'].strip()
        answer = sampled_answers.iloc[index]['answer'].strip()
        
        similarity_score = compute_similarity(model, tokenizer, question, answer)
        
        results.append({
            'model': model_name,
            'question': question,
            'answer': answer,
            'similarity_score': similarity_score
        })
        
        similarities.append(similarity_score)

    # Top-1 ve Top-5 hesaplama
    top_1 = np.max(similarities)
    top_5 = np.sort(similarities)[-5:].mean()  # En yüksek 5 benzerliğin ortalaması

    print(f"{model_name} - Top-1: {top_1:.4f}, Top-5: {top_5:.4f}")

# Sonuçları DataFrame'e çevirme
results_df = pd.DataFrame(results)

# Sonuçları kaydetme
results_output_path = 'data/results/enrde_similarity_results.csv'
results_df.to_csv(results_output_path, index=False, encoding='utf-8-sig', sep=';')

print("Sonuçlar kaydedildi:", results_output_path)
