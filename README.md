# Project: Evaluation of Language Models (T5 and DeepSeek-Coder) for Programming Question Generation and Summarization

## 🎯 Project Goal
This project evaluates the performance of two powerful language models—T5 and DeepSeek-Coder—on Stack Overflow questions. The main objective is to assess the models' ability to summarize questions, generate answers, and compare them against actual data using ROUGE and correlation with user scores.

---

## 🛠 Workflow

### 1. Data Preparation
- Raw data was loaded from `QueryResults.csv`, including question title, body, accepted answer, tags, and user score.
- HTML tags and special characters were cleaned using regular expressions.

### 2. Summarization Using T5
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize(text):
    return summarizer("summarize: " + text, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
```

---

### 3. Answer Generation Using DeepSeek-Coder
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype=torch.bfloat16).cuda()

def generate_answer(text):
    inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
```

---

### 4. Evaluation Using ROUGE
```python
from rouge import Rouge
rouge = Rouge()

score = rouge.get_scores(predicted, actual)[0]['rouge-1']['f']
```

---

## 📊 Statistical Analysis
- ROUGE scores between generated summaries/answers and actual data were computed.
- Pearson correlation was calculated between ROUGE scores and user-assigned Stack Overflow scores.
- T5 showed weak positive correlation for summarization (~0.11).
- DeepSeek-Coder showed stronger performance in code-related text but slightly negative correlation for answer quality.
- Best correlation (~0.36) was seen in DeepSeek’s readability of code.

---

## 📌 Conclusion
- Language models can generate relevant summaries and responses.
- ROUGE alone is not sufficient to assess human-like quality.
- Future improvements should involve semantic metrics and human evaluation.

---

## 📁 Suggested Project Structure

```
📁 title-quality-project/
│
├── README.md
├── data/
│   └── QueryResults.csv
├── notebooks/
│   ├── t5_summarization_eval.ipynb
│   └── deepseek_generation_eval.ipynb
├── outputs/
│   └── rouge_scores.csv
```