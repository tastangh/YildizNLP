from transformers import AutoTokenizer, AutoModel
import torch

# Kullanılacak modeller
model_names = [
    "intfloat/multilingual-e5-large-instruct",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "Alibaba-NLP/gte-multilingual-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "dbmdz/bert-base-turkish-cased"
]

# Modelleri ve tokenları yükle
models = {}
tokenizers = {}
for model_name in model_names:
    try:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        models[model_name] = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"{model_name} yüklendi.")
    except Exception as e:
        print(f"{model_name} yüklenirken hata oluştu: {e}")

# Yüklenen modellerin boyutlarını kontrol et
for model_name in model_names:
    if model_name in models:
        print(f"{model_name} - Model Boyutu: {models[model_name].num_parameters()} parametre")
