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
question_answer_path = 'data/results/sampled_question_answer_for_question.csv'
data = pd.read_csv(question_answer_path, sep=';', header=0)
data.columns = ['index', 'question', 'answer']

# Soruları ve cevapları ayır
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Veriyi train, validation ve test setlerine ayır
train_questions, temp_questions, train_answers, temp_answers = train_test_split(
    questions, answers, test_size=0.2, random_state=42)
val_questions, test_questions, val_answers, test_answers = train_test_split(
    temp_questions, temp_answers, test_size=0.5, random_state=42)

# Temsil verilerini çıkartma
def get_representation(model_data, texts):
    tokenizer, model = model_data
    representations = []
    for text in tqdm(texts, desc="Tokenizing texts", leave=False):  # İlerleme çubuğu ekle
        # Tokenize the text without altering the data types
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Ensure that the input is on the correct device without changing its type
            outputs = model(**inputs)

            # Çıktıyı Float32'e çevir
            last_hidden_state = outputs.last_hidden_state.float()

        # Use the mean of the last hidden state to represent the input
        representations.append(last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(representations)

#Model Eğitme
def train_model(model_data, train_questions, train_answers, val_questions, val_answers, epochs=5, lr=1e-5):
    tokenizer, model = model_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_question_reps = get_representation(model_data, train_questions)
    train_answer_reps = get_representation(model_data, train_answers)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Cosine similarity
        similarities = cosine_similarity(train_question_reps, train_answer_reps)
        loss = 1 - torch.tensor(similarities, requires_grad=True).mean()  # Cosine similarity loss
        loss.backward()  # PyTorch tensor üzerinde backward çağrısı
        optimizer.step()

        # Evaluate on validation set
        val_question_reps = get_representation(model_data, val_questions)
        val_answer_reps = get_representation(model_data, val_answers)
        val_similarities = cosine_similarity(val_question_reps, val_answer_reps)
        
        val_loss = 1 - torch.tensor(val_similarities).mean()  # Validation loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")


# Başarıları değerlendirme
def evaluate_model(model_name):
    model_data = load_model(model_name)
    train_model(model_data, train_questions, train_answers, val_questions, val_answers)

    # Final evaluation on the test set
    question_reps = get_representation(model_data, test_questions)
    answer_reps = get_representation(model_data, test_answers)
    similarities = cosine_similarity(question_reps, answer_reps)

    top1_success = 0
    top5_success = 0

    for i in range(len(test_questions)):
        top_indices = np.argsort(similarities[i])[-5:]
        if top_indices[-1] == i:
            top1_success += 1
        if i in top_indices:
            top5_success += 1

    total_questions = len(test_questions)
    
    top1_percentage = (top1_success / total_questions) * 100
    top5_percentage = (top5_success / total_questions) * 100

    return model_name, top1_percentage, top5_percentage

# Ana işlem
if __name__ == '__main__':
    set_start_method('spawn')

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_model, model_names), total=len(model_names), desc="Evaluating models"))

    with open("model_success_results.txt", "w") as f:
        f.write("Model Adı | Top 1 Başarı (%) | Top 5 Başarı (%)\n")
        f.write("-----------------------------------------\n")
        for model_name, top1, top5 in results:
            f.write(f"{model_name} | {top1:.2f} | {top5:.2f}\n")

    print("Sonuçlar model_success_results.txt dosyasına yazıldı.")

# t-SNE uygulama ve görselleştirme
for model_name in model_names:
    model_data = load_model(model_name)
    # Tüm temsil verilerini al
    all_representations = np.vstack([get_representation(model_data, questions)] + 
                                     [get_representation(model_data, [answer]) for answer in answers])
    
    # t-SNE uygulama
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_representations)

    plt.figure(figsize=(10, 6))
    
    # Soru ve cevapları tek renk ile göstermek
    plt.scatter(tsne_results[:len(questions), 0], tsne_results[:len(questions), 1], 
                label='Soru', color='blue', alpha=0.6)  # Soru için mavi
    plt.scatter(tsne_results[len(questions):, 0], tsne_results[len(questions):, 1], 
                label='Cevap', color='orange', alpha=0.6)  # Cevap için turuncu
    
    plt.title(f"{model_name} - t-SNE Görselleştirme")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid()
    plt.show()