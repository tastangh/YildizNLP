import pandas as pd

# CSV dosyalarının yolları
question_answer_path = 'data/results/sampled_question_answer.csv'

# CSV dosyalarını oku
questions_df = pd.read_csv(question_answer_path,sep=';')

# İlk birkaç satırı görüntüle
print(questions_df.head())

