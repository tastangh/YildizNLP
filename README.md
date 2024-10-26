
In this project, we utilize six models to measure similarity between pairs of texts. The selected models meet the following criteria:
- **Multilingual**
- **Less than 1 billion parameters**

### Models

1. **DistilBERT (distilbert-base-multilingual-cased)**
   - **Rank**: 14 in the MTEB leaderboard
   - **Parameters**: 134 million
   - **Multilingual**: Yes

2. **mBERT (bert-base-multilingual-cased)**
   - **Rank**: 15 in the MTEB leaderboard
   - **Parameters**: 110 million
   - **Multilingual**: Yes

3. **XLM-RoBERTa (xlm-roberta-base)**
   - **Rank**: 7 in the MTEB leaderboard
   - **Parameters**: 125 million
   - **Multilingual**: Yes

4. **mT5 (mt5-small)**
   - **Rank**: 9 in the MTEB leaderboard
   - **Parameters**: 300 million
   - **Multilingual**: Yes

5. **T5 (t5-small)**
   - **Rank**: 21 in the MTEB leaderboard
   - **Parameters**: 60 million
   - **Multilingual**: Limited (adaptable for multilingual tasks)

These models are chosen based on their performance in multilingual text processing tasks, ensuring effective and efficient similarity measurement while adhering to the specified criteria.

For further details on the models and their performance, please visit the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
