from datasets import load_dataset
import pandas as pd

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
new_df = df[['index','question', 'answer']]

# CSV olarak kaydetme (utf-8-sig ile)
new_df.to_csv("question_answer.csv", index=False, encoding='utf-8-sig', sep=';')

# İlk birkaç satırı görüntüleme (isteğe bağlı)
print(new_df.head())
