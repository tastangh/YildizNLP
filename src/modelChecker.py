from transformers import AutoTokenizer, AutoModel
import torch

# Kullanılacak modeller
model_names = [
    "sentence-transformers/all-MiniLM-L12-v2",
    "jinaai/jina-embeddings-v3",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
    "thenlper/gte-large",
    "nomic-ai/nomic-embed-text-v1",
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
