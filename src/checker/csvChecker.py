import pandas as pd

# CSV dosyalarının yolları
question_answer_path = 'data/results/sampled_question_answer_for_question.csv'
answer_question_path = 'data/results/sampled_question_answer_for_answer.csv'

# CSV dosyalarını oku
questions_df = pd.read_csv(question_answer_path,sep=';')
answers_df = pd.read_csv(answer_question_path,sep=';')

# İlk birkaç satırı görüntüle
print(questions_df.head())
print(answers_df.head())

