import pandas as pd
import random
import os

# Veri yolu
data_path = 'data/processed/processed_questions_answers.csv'

# Veriyi yükle
data = pd.read_csv(data_path, sep=';')

# Sütun adlarındaki boşlukları temizleme
data.columns = data.columns.str.strip()

# Sonuçların kaydedileceği dizin
results_dir = 'data/results/'
os.makedirs(results_dir, exist_ok=True)

# Rastgele 1000 soru ve cevap seçimi
def process_data(data):
    # Rastgele 1000 soru seç
    sampled_questions = data.sample(1000, random_state=32)

    # Soru ve cevapları içeren dosyayı kaydet
    questions_output_file_path = os.path.join(results_dir, "sampled_question_answer.csv")
    sampled_questions.to_csv(questions_output_file_path, index=False, encoding='utf-8-sig', sep=';')
    print("Rastgele 1000 soru ve cevap işlendi ve kaydedildi:")
    print("Soru dosyası:", questions_output_file_path)

# Ana işlev
def main():
    process_data(data)

if __name__ == "__main__":
    main()
