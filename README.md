# ABTE-Sentiment-DL

> **EN / VI bilingual README** — Full project README in both **English** and **Tiếng Việt**.  
> Quick language index:
>
> - [English](#english)
> - [Tiếng Việt](#tiếng-việt)

---

## English

### Overview
**ABTE-Sentiment-DL** implements a complete pipeline for:
- **Aspect-Based Term Extraction (ABTE)** at token level with BIO tags (`B-Term`, `I-Term`, `O`).
- **Sentence-level Sentiment Classification** with 3 classes: **Negative / Neutral / Positive**.

Supported models:
- 🧱 **CNN 1D (Conv1D)**
- 🌀 **LSTM**
- 🔷 **Transformer Encoder (custom)**
- 🤗 **Fine-tune DistilBERT** for **TokenClassification** (ABTE)

A **Streamlit demo** is included. It auto-detects `.safetensors` checkpoints and highlights aspects + predicts sentence/aspect polarities.

---

### Datasets

| Dataset            | Task           | Classes | Train   | Test  |
|--------------------|----------------|:-------:|--------:|------:|
| ABTE-Restaurants   | ABTE (BIO)     |   3     |   3,602 | 1,119 |
| Sentiment-Tweets   | Sentiment (3c) |   3     |  40,000 | 10,000 |

- **ABTE**: columns `Tokens` (`list[str]`), `Tags` (`list[int]` with `O=0, B=1, I=2`).
- **Sentiment**: columns `text`, `label` (`0/1/2` ⇔ `Neg/Neu/Pos`).

> Replace with your sources/links if needed.

---

### Settings
Default unless specified:
- **Epochs**: 20 (**Early Stopping** `patience=3`)
- **Batch size**: 256 (train/eval)
- **Weight decay**: 0.01
- **Optimizers & LR**
  - CNN/LSTM/Transformer: `Adam(lr=2e-3)`
  - DistilBERT fine-tuning: `AdamW(lr=5e-5)`
- **Tokenizer**: WordLevel (or BPE), trained on train split if applicable
- **Checkpoint**: `save_safetensors=True` → produces `model.safetensors`
- **1 epoch** = full pass over data (each sample used exactly once)

---

### Installation
```bash
# 1) Create venv
python -m venv .venv
# 2) Activate
#   Windows PowerShell:
.venv\Scripts\activate
#   Linux/macOS:
# source .venv/bin/activate
# 3) Install deps
pip install -U -r requirements.txt
```

**requirements.txt (suggested)**
```
streamlit>=1.37
transformers>=4.41
tokenizers>=0.15
torch>=2.0
evaluate>=0.4
seqeval==1.2.2
```

---

## Training

### Quick commands (per model)
```bash
# CNN 1D
python train.py --model conv1d        --output_dir out_cnn

# LSTM
python train.py --model lstm          --output_dir out_lstm

# Transformer encoder (custom)
python train.py --model transformer   --output_dir out_transformer

# DistilBERT fine-tuning for ABTE
python train.py --model finetune      --output_dir out_finetune \
  --pretrained distilbert/distilbert-base-uncased
```

### Custom parameters
```bash
python train.py --model conv1d \
  --epochs 20 --batch_size 256 --lr 2e-3 --patience 3 \
  --output_dir out_cnn
```

> If you only have **one CSV** for sentiment, the script splits 80/20 with `stratify=label`.

---

## Experiments
Hardware examples:
- **GPU**: RTX 4090 24GB / TITAN X 12GB
- **VRAM guide**: CNN/LSTM/Transformer ~ 1–3 GB; DistilBERT fine-tune ~ 6–8 GB

**Example test results (replace with your numbers):**

| Task | Size  | Model                 | F1 (reproduced) | Ref |
|------|-------|-----------------------|----------------:|----:|
| ABTE | Small | CNN 1D                | 0.xx            |  –  |
| ABTE | Small | LSTM                  | 0.xx            |  –  |
| ABTE | Small | Transformer Encoder   | 0.xx            |  –  |
| ABTE | Small | **DistilBERT (ft)**   | **0.xx**        |  –  |

- Detailed config/logs per run: `out_<model>/logs.txt`
- Checkpoints/state: `out_<model>/checkpoint-XXXX/`

---

## Curves (F1 / Loss)

Generate curves from `trainer_state.json` (collected in each `out_<model>/checkpoint-*/`):
```bash
python draw.py
```

Artifacts produced:
- `compare_f1.png` — evaluation **F1** vs. epoch  
- `compare_loss.png` — evaluation **loss** vs. epoch

Embed in README after commit:
```md
![F1 curves](./compare_f1.png)
![Loss curves](./compare_loss.png)
```

---

## Streamlit Demo

Run the demo:
```bash
streamlit run app.py
```

Notes:
- The app **auto-detects** the latest `model.safetensors` (prioritizes `out_finetune/checkpoint-*`).
- To pin a directory manually, set inside **app.py**:
```python
ABTE_DIR = "model_abte"  # folder containing model.safetensors + tokenizer files
```

Minimum files needed for inference (keep these in the model folder):
- `model.safetensors`, `config.json`
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- (`vocab.txt` / `merges.txt` / `spiece.model` depending on tokenizer)

---

## Project Structure
```
ABTE-Sentiment-DL/
├─ app.py
├─ train.py
├─ models.py
├─ init.py
├─ draw.py
├─ requirements.txt
├─ out_cnn/
├─ out_lstm/
├─ out_transformer/
├─ out_finetune/
└─ model_abte/
   ├─ model.safetensors
   ├─ config.json
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ special_tokens_map.json
   └─ (vocab.txt | merges.txt | spiece.model)
```

---

## Deployment Notes
- **Security**: use `safetensors` for checkpoints and `use_safetensors=True` for loading.
- **Large Git pushes (HTTP 408)**: migrate large files to **Git LFS**  
  `git lfs migrate import --include="*.safetensors,*.pt,*.pth,*.bin,*.h5" --everything`  
  or host the model on **Hugging Face Hub** and keep GitHub code-only.

**.gitignore (suggested)**
```
__pycache__/
.venv/
**/optimizer.pt
**/scheduler.pt
**/scaler.pt
**/rng_state.pth
**/trainer_state.json
**/training_args.bin
**/pytorch_model.bin
**/*.pt
!**/model.safetensors
```

---

### License
MIT or Apache-2.0 (choose one and include a LICENSE file).

### Acknowledgements
Hugging Face (Transformers/Datasets/Evaluate), SeqEval, and the public datasets used.

---

## Tiếng Việt

### Giới thiệu
**ABTE-Sentiment-DL** hiện thực trọn vẹn:
- **Trích aspect (ABTE)** ở mức token (nhãn BIO: `B-Term`, `I-Term`, `O`).
- **Phân loại cảm xúc câu** với 3 lớp: **Tiêu cực / Trung tính / Tích cực**.

Hỗ trợ các mô hình:
- 🧱 **CNN 1D (Conv1D)**
- 🌀 **LSTM**
- 🔷 **Transformer Encoder (tùy biến)**
- 🤗 **Fine-tune DistilBERT** cho **TokenClassification** (ABTE)

Có **demo Streamlit** tự dò checkpoint `.safetensors`, tô màu aspect và dự đoán cảm xúc câu/cho từng aspect.

---

### Dữ liệu

| Dataset            | Bài toán      | Số lớp | Train   | Test  |
|--------------------|---------------|:------:|--------:|------:|
| ABTE-Restaurants   | ABTE (BIO)    |   3    |   3,602 | 1,119 |
| Sentiment-Tweets   | Cảm xúc (3c)  |   3    |  40,000 | 10,000 |

- **ABTE**: cột `Tokens` (`list[str]`), `Tags` (`list[int]` với `O=0, B=1, I=2`).
- **Sentiment**: cột `text`, `label` (`0/1/2` ⇔ `Tiêu cực/Trung tính/Tích cực`).

---

### Thiết lập
Mặc định (trừ khi nêu khác):
- **Epochs**: 20 (**Early Stopping** `patience=3`)
- **Batch**: 256 (train/eval)
- **Weight decay**: 0.01
- **Tối ưu & LR**
  - CNN/LSTM/Transformer: `Adam(lr=2e-3)`
  - Fine-tune DistilBERT: `AdamW(lr=5e-5)`
- **Tokenizer**: WordLevel (hoặc BPE)
- **Checkpoint**: `save_safetensors=True` → sinh `model.safetensors`
- **1 epoch** = duyệt hết dữ liệu (mỗi mẫu dùng đúng 1 lần)

---

## Huấn luyện

### Lệnh nhanh theo mô hình
```bash
# CNN 1D
python train.py --model conv1d        --output_dir out_cnn

# LSTM
python train.py --model lstm          --output_dir out_lstm

# Transformer encoder
python train.py --model transformer   --output_dir out_transformer

# Fine-tune DistilBERT cho ABTE
python train.py --model finetune      --output_dir out_finetune \
  --pretrained distilbert/distilbert-base-uncased
```

### Tuỳ biến tham số
```bash
python train.py --model conv1d \
  --epochs 20 --batch_size 256 --lr 2e-3 --patience 3 \
  --output_dir out_cnn
```

> Nếu chỉ có **1 file CSV** cho sentiment: script sẽ tách `train/test` 80/20 với `stratify=label`.

---

## Thực nghiệm
Phần cứng:
- **GPU**: RTX 4090 24GB / TITAN X 12GB
- **VRAM tham khảo**: CNN/LSTM/Transformer ~ 1–3 GB; Fine-tune DistilBERT ~ 6–8 GB

**Kết quả test (thay bằng số thực tế):**

| Bài toán | Size  | Mô hình               | F1 (tái lập) | Tham chiếu |
|----------|-------|-----------------------|-------------:|-----------:|
| ABTE     | Small | CNN 1D                | 0.xx         |     –      |
| ABTE     | Small | LSTM                  | 0.xx         |     –      |
| ABTE     | Small | Transformer Encoder   | 0.xx         |     –      |
| ABTE     | Small | **DistilBERT (ft)**   | **0.xx**     |     –      |

- Cấu hình/log chi tiết: `out_<model>/logs.txt`  
- Checkpoint/state: `out_<model>/checkpoint-XXXX/`

---

## Biểu đồ (F1 / Loss)

Sinh từ `trainer_state.json` (nằm trong từng `out_<model>/checkpoint-*/`):
```bash
python draw.py
```

File kết quả:
- `compare_f1.png` — **F1** theo epoch  
- `compare_loss.png` — **loss** theo epoch

Chèn vào README:
```md
![F1 curves](./compare_f1.png)
![Loss curves](./compare_loss.png)
```

---

## Demo Streamlit

Chạy demo:
```bash
streamlit run app.py
```

Lưu ý:
- App **tự dò** `model.safetensors` mới nhất (ưu tiên `out_finetune/checkpoint-*`).
- Cố định thư mục model trong **app.py**:
```python
ABTE_DIR = "model_abte"  # chứa model.safetensors + tokenizer
```

Gói tối thiểu cho suy luận (giữ trong thư mục model):
- `model.safetensors`, `config.json`
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- (`vocab.txt` / `merges.txt` / `spiece.model` tuỳ tokenizer)

---

## Cấu trúc dự án
```
ABTE-Sentiment-DL/
├─ app.py
├─ train.py
├─ models.py
├─ init.py
├─ draw.py
├─ requirements.txt
├─ out_cnn/
├─ out_lstm/
├─ out_transformer/
├─ out_finetune/
└─ model_abte/
   ├─ model.safetensors
   ├─ config.json
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ special_tokens_map.json
   └─ (vocab.txt | merges.txt | spiece.model)
```

---

## Ghi chú deploy
- **Bảo mật**: dùng `safetensors` & `use_safetensors=True` khi load.
- **Repo nặng / push lỗi 408**: chuyển lịch sử file lớn sang **Git LFS**  
  `git lfs migrate import --include="*.safetensors,*.pt,*.pth,*.bin,*.h5" --everything`  
  hoặc đưa model lên **Hugging Face Hub** để GitHub chỉ chứa code.

**.gitignore (gợi ý)**
```
__pycache__/
.venv/
**/optimizer.pt
**/scheduler.pt
**/scaler.pt
**/rng_state.pth
**/trainer_state.json
**/training_args.bin
**/pytorch_model.bin
**/*.pt
!**/model.safetensors
```

---

### Giấy phép
MIT hoặc Apache-2.0 (chọn và thêm file `LICENSE`).

### Ghi công
Hugging Face (Transformers/Datasets/Evaluate), SeqEval và các bộ dữ liệu công khai bạn sử dụng.
