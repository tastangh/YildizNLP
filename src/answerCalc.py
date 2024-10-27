import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging

# Günlükleme ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Kullanılacak modeller
model_names = [
    "intfloat/multilingual-e5-large-instruct",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "Alibaba-NLP/gte-multilingual-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "dbmdz/bert-base-turkish-cased"
]

# CSV dosyası
answer_question_path = 'data/results/sampled_question_answer_for_answer.csv'

# CSV dosyasını oku
logging.info("CSV dosyası okunuyor...")
answers_df = pd.read_csv(answer_question_path, sep=';')
logging.info("CSV dosyası başarıyla okundu.")

# Cevapları ve soruları ayırma
answers = answers_df['answer'].tolist()
questions = answers_df['question'].tolist()  # Sorular cevaplardan alınıyor

# Rastgele 1000 cevap ve soruyu seç
np.random.seed(42)  # Tekrar üretilebilirlik için sabit rastgelelik
random_indices = np.random.choice(len(answers), size=1000, replace=False)
selected_answers = [answers[i] for i in random_indices]
selected_questions = [questions[i] for i in random_indices]

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

# Fonksiyon: Temsilleri al
def get_embeddings(texts, model_name):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    embeddings = []

    logging.info(f"{model_name} için temsilleri alınıyor...")
    for text in tqdm(texts, desc=f"{model_name} için temsilleri alınıyor...", leave=False):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))  # Temsil olarak ortalama alınır

    logging.info(f"{model_name} için temsiller alındı.")
    return torch.vstack(embeddings)  # Tüm temsilleri birleştir

# Cevapların temsillerini al
answer_embeddings = {}
for model_name in model_names:
    answer_embeddings[model_name] = get_embeddings(selected_answers, model_name)

# Soruların temsillerini al
question_embeddings = {}
for model_name in model_names:
    question_embeddings[model_name] = get_embeddings(selected_questions, model_name)

# Benzerlik hesaplama ve en benzer 5 soruyu bulma
top_1_results = {}
top_5_results = {}

# Cevaplardan sorulara
logging.info("Benzerlik hesaplanıyor ve en benzer 5 soru bulunuyor...")
for model_name in model_names:
    similarities = cosine_similarity(answer_embeddings[model_name], question_embeddings[model_name])
    
    # Her bir cevap için en benzer 5 soruyu bul
    top_5_indices = np.argsort(similarities, axis=1)[:, -5:]  # En yüksek 5 benzerlik indeksini al
    top_5_results[model_name] = top_5_indices
    
    # Top 1 başarı
    top_1_indices = np.argmax(similarities, axis=1)
    top_1_results[model_name] = top_1_indices

# Gerçek soruların indeksini belirle
true_questions = selected_questions  # Gerçek sorular doğrudan cevaplardan alınır

top_1_accuracy = {}
top_5_accuracy = {}

# Başarı oranlarını hesapla
logging.info("Başarı oranları hesaplanıyor...")
for model_name in model_names:
    top_1_correct = np.sum(np.array(true_questions)[top_1_results[model_name]] == true_questions)
    top_5_correct = np.sum([1 if true_questions[i] in np.array(true_questions)[top_5_results[model_name][i]] else 0 for i in tqdm(range(len(top_5_results[model_name])), desc=f"{model_name} için Top 5 doğru sayımı", leave=False)])
    
    top_1_accuracy[model_name] = top_1_correct / len(selected_questions) * 100  # Yüzde olarak başarı
    top_5_accuracy[model_name] = top_5_correct / len(selected_questions) * 100  # Yüzde olarak başarı

# Başarı oranlarını yazdır
for model_name in model_names:
    logging.info(f"{model_name} - Top 1 Başarı: {top_1_accuracy[model_name]:.2f}%")
    logging.info(f"{model_name} - Top 5 Başarı: {top_5_accuracy[model_name]:.2f}%")

# Görselleştirme
for model_name in model_names:
    # TSNE uygulama
    logging.info(f"{model_name} için TSNE uygulanıyor...")
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat((answer_embeddings[model_name], question_embeddings[model_name]), dim=0)
    tsne_results = tsne.fit_transform(all_embeddings)

    # Görselleştirme
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:len(selected_questions), 0], tsne_results[:len(selected_questions), 1], label='Questions', color='blue', alpha=0.5)
    plt.scatter(tsne_results[len(selected_questions):, 0], tsne_results[len(selected_questions):, 1], label='Answers', color='red', alpha=0.5)
    plt.title(f'TSNE Visualization for {model_name}')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.show()
    logging.info(f"{model_name} için TSNE görselleştirmesi tamamlandı.")
