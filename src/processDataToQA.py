import pandas as pd
from datasets import load_dataset
import os

# Hugging Face'deki veri setini yükle
dataset = load_dataset("merve/turkish_instructions")

# Eğitim verisini DataFrame'e çevirme
df = pd.DataFrame(dataset['train'])

# Talimat ve giriş sütunlarını birleştirerek yeni bir 'soru' sütunu oluşturma
# Boş değerleri doldurmak için fillna kullanarak birleştir
df['question'] = df['talimat'].fillna('') + ' ' + df[' giriş'].fillna('')
df['answer'] = df[' çıktı']
df['index'] = df['Unnamed: 0']

# 'soru' ve 'çıktı' sütunlarını içeren yeni bir DataFrame oluşturma
new_df = df[['index', 'question', 'answer']]

# Çıktı dosyasının kaydedileceği yol
output_dir = 'data/processed/'
os.makedirs(output_dir, exist_ok=True)  # Klasörü oluştur
output_file_path = os.path.join(output_dir, "processed_questions_answers.csv")

# Yeni DataFrame'i CSV dosyasına kaydetme
new_df.to_csv(output_file_path, index=False, encoding='utf-8-sig', sep=';')

# İlk birkaç satırı görüntüleme (isteğe bağlı)
print(new_df.head())
