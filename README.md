# NLP Project

## Models Used

The following models will be utilized to create text embeddings. They are ranked according to their performance based on the MTEB leaderboard:

1. **[intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)**
   - **Description:** A multilingual model designed for instruction-based tasks, effective in generating contextual embeddings across languages.

2. **[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)**
   - **Description:** A model optimized for generating embeddings that excel in retrieval tasks and can handle diverse queries efficiently.

3. **[thenlper/gte-large](https://huggingface.co/thenlper/gte-large)**
   - **Description:** A large model trained on multiple tasks, offering high-quality embeddings suitable for various NLP applications.

4. **[nomic-ai/nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1)**
   - **Description:** A text embedding model that focuses on generating embeddings for text data with a wide range of applications.

5. **[sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)**
   - **Description:** A compact model designed for sentence embeddings, providing fast inference and good performance on semantic similarity tasks.

6. **[jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)**
   - **Description:** A model built for creating embeddings from text data, particularly effective in the Jina AI ecosystem.

7. **[dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)**
   - **Description:** A Turkish BERT model that is fine-tuned for various Turkish NLP tasks, suitable for generating embeddings for Turkish text.

## Requirements

To run this project, you will need to install the following Python packages. Make sure to install `einops` before the others, as it is required for the **jinaai/jina-embeddings-v3** and **nomic-ai/nomic-embed-text-v1** models:

```bash
pip install einops
pip install pandas numpy scikit-learn transformers torch matplotlib tqdm