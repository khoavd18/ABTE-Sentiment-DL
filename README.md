# ABTE-Sentiment-DL

Aspect-Based Term Extraction (B/I/O) & Sentiment (Negative / Neutral / Positive) với các mô hình:
- 🧱 **CNN 1D (Conv1D)**
- 🌀 **LSTM**
- 🔷 **Transformer Encoder**
- 🤗 **Fine-tune DistilBERT (TokenClassification)**

Kèm **demo Streamlit**: tự dò checkpoint `.safetensors` (không cần Torch ≥ 2.6 nhờ `use_safetensors=True`).

---

## Introduction

Hiện thực PyTorch cho:
- **ABTE (token-level)**: gán nhãn `B-Term / I-Term / O` để trích **aspect terms**.
- **Sentiment (sentence-level)**: phân loại **Negative / Neutral / Positive** cho câu; đồng thời **ước lượng polarity** cho từng aspect theo ngữ cảnh cục bộ xung quanh span.

> Demo Streamlit tô màu aspect terms, hiển thị polarity cho câu & từng aspect.

---

## Datasets

| Dataset            | Task       | Classes | Train  | Test  |
|--------------------|------------|:-------:|-------:|------:|
| ABTE-Restaurants   | ABTE (BIO) |   3     | 3,602  | 1,119 |
| Sentiment-Tweets   | Sentiment  |   3     | 40,000 | 10,000 |

- **ABTE**: cột `Tokens` (`list[str]`) & `Tags` (`list[int]` với `O=0, B=1, I=2`).
- **Sentiment**: cột `text`, `label` (`0/1/2` ⇔ `Neg/Neu/Pos`).

> Thay link/nguồn dataset của bạn tại đây: `<HuggingFace/Kaggle/local path>`.

---

## Setting

- **Epochs:** 20  (**Early Stopping** `patience=3`)
- **Batch size:** 256 (train/eval)
- **Weight decay:** 0.01
- **Optimizer & LR**
  - CNN/LSTM/Transformer: `Adam(lr=2e-3)`
  - Fine-tune DistilBERT: `AdamW(lr=5e-5)`
- **Tokenizer:** WordLevel (hoặc BPE)
- **Checkpoint:** `save_safetensors=True` → sinh `model.safetensors`
- **1 epoch** = duyệt hết dữ liệu (mỗi mẫu xuất hiện đúng 1 lần)

---

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\activate
pip install -U -r requirements.txt
