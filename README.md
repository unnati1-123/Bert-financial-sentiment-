#  Financial Sentiment Analysis — BERT

A fine-tuned BERT model that classifies financial text into **Negative**, **Neutral**, or **Positive** sentiment with **96.48% accuracy**.

Built as part of an NLP + Finance course project exploring algorithmic trading signals from financial tweets.

---

## Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 84.14% |
| LSTM | 85.46% |
| **BERT (this model)** | **96.48%**  |

---

##  Labels

| Label | Meaning |
|-------|---------|
| LABEL_0 | Negative 📉 |
| LABEL_1 | Neutral ➡️ |
| LABEL_2 | Positive 📈 |

---

##  Model on Hugging Face

The trained model weights are hosted on Hugging Face:

 **[STUDENT-135080/NLP_](https://huggingface.co/STUDENT-135080/NLP_)**

---

##  Quick Start

### Install dependencies
```bash
pip install transformers torch safetensors
```

### Run the app
```bash
python app.py
```

### Use in your own code
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="STUDENT-135080/NLP_"
)

result = classifier("Apple stock surges after record earnings beat")
print(result)
# [{'label': 'LABEL_2', 'score': 0.97}]  →  Positive 📈
```

---

## Repository Structure

```
bert-financial-sentiment/
├── app.py          # Run the model locally
└── README.md       # This file
```

---

##  Team

unnati

---

## 📚 References

- Araci, Dogu. (2019). *FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models.* ArXiv:1908.10063
- Bhuiyan et al. (2025). *Deep Learning for Algorithmic Trading.* Array, vol. 26.
