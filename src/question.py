import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm  # Ilerleme çubuğu için tqdm'i içe aktarın

# Veri dosyasını yükleyin
question_answer_path = 'data/results/sampled_question_answer_for_question.csv'
df = pd.read_csv(question_answer_path, delimiter=';')

# Modeli ve tokenizer'ı yükleyin
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Temsilleri hesaplama fonksiyonu
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()  # Temsil vektörü

# Cevapların temsillerini hesaplayın
df['embeddings'] = df['answer'].apply(get_embeddings)

# Kosinüs benzerliğini hesaplama fonksiyonu
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Benzer cevapları bulma fonksiyonu
def find_similar_answers(question_embedding, all_embeddings, top_k=5):
    similarities = []
    for emb in all_embeddings:
        sim = cosine_similarity(question_embedding, emb.numpy())
        similarities.append(sim)
    return np.argsort(similarities)[-top_k:][::-1]  # En benzer top_k indekslerini döndür

# Başarı hesaplama fonksiyonu
def calculate_success(similar_indices, true_answer_index):
    top1_success = 1 if similar_indices[0] == true_answer_index else 0
    top5_success = 1 if true_answer_index in similar_indices else 0
    return top1_success, top5_success

# Genel başarı hesaplaması
total_top1 = 0
total_top5 = 0
total_questions = len(df)

# tqdm ile ilerleme çubuğu ekleyin
for i in tqdm(range(total_questions), desc="Processing Questions"):
    question_embedding = get_embeddings(df['question'][i])  # Soru temsili
    similar_indices = find_similar_answers(question_embedding, df['embeddings'].tolist())
    true_answer_index = i  # Doğru cevap indeksi
    top1, top5 = calculate_success(similar_indices, true_answer_index)
    total_top1 += top1
    total_top5 += top5

# Genel başarı oranlarını hesaplayın
top1_success_rate = total_top1 / total_questions
top5_success_rate = total_top5 / total_questions
print(f"Overall Top 1 Success Rate: {top1_success_rate:.2f}, Overall Top 5 Success Rate: {top5_success_rate:.2f}")
