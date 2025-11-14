# app.py — chạy thẳng: streamlit run app.py
import os, glob, re
from typing import List, Tuple

import torch
import torch.nn.functional as F
import streamlit as st
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,     # ABTE (token-level)
    AutoTokenizer as AutoTok2, AutoModelForSequenceClassification  # sentiment (sentence-level)
)

# =========================
# Auto-detect checkpoint ABTE (ưu tiên out_finetune/checkpoint-*)
# =========================
def find_abte_checkpoint() -> str | None:
    # 1) Ưu tiên: out_finetune/checkpoint-*/model.safetensors
    cands = sorted(glob.glob("output/out_finetune/checkpoint-*/model*.safetensors"))
    if cands:
        # lấy checkpoint có step lớn nhất
        def step_dir(p):  # .../checkpoint-1234/model.safetensors -> 1234
            m = re.search(r"checkpoint-(\d+)", p)
            return int(m.group(1)) if m else -1
        best = max(cands, key=step_dir)
        return os.path.dirname(best)  # trả về thư mục checkpoint

    # 2) Nếu không có, tìm bất kỳ thư mục con nào chứa model.safetensors
    any_safetensors = sorted(glob.glob("**/model*.safetensors", recursive=True))
    if any_safetensors:
        # ưu tiên thư mục có config.json + tokenizer.json
        def score_dir(p):
            d = os.path.dirname(p)
            s = 0
            if os.path.isfile(os.path.join(d, "config.json")): s += 1
            if os.path.isfile(os.path.join(d, "tokenizer.json")): s += 1
            if os.path.isfile(os.path.join(d, "special_tokens_map.json")): s += 1
            return s
        best = max(any_safetensors, key=score_dir)
        return os.path.dirname(best)

    # 3) Không có safetensors nào
    return None

# =========================
# Cấu hình mặc định
# =========================
ABTE_DIR = find_abte_checkpoint()  # Tự dò
SENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # có sẵn safetensors
SENT_LABELS = ["Negative", "Neutral", "Positive"]

ID2LABEL = {0: "O", 1: "B-Term", 2: "I-Term"}
LABEL2ID = {"O": 0, "B-Term": 1, "I-Term": 2}

# =========================
# UI helpers
# =========================
def color_tag(label: str) -> str | None:
    if label == "B-Term": return "#ffdf80"  # vàng nhạt
    if label == "I-Term": return "#ffd1dc"  # hồng nhạt
    return None

def label_badge(label: str) -> str:
    color = {"B-Term": "#ffbf00", "I-Term": "#ff80ab", "O": "#e0e0e0"}
    c = color.get(label, "#e0e0e0")
    return f"<span style='background:{c}; padding:1px 6px; border-radius:10px; font-size:12px; margin-left:4px;'>{label}</span>"

def extract_spans(words: List[str], labels: List[str]) -> List[str]:
    spans, cur = [], []
    for w, lb in zip(words, labels):
        if lb == "B-Term":
            if cur: spans.append(" ".join(cur)); cur = []
            cur = [w]
        elif lb == "I-Term" and cur:
            cur.append(w)
        else:
            if cur: spans.append(" ".join(cur)); cur = []
    if cur: spans.append(" ".join(cur))
    return spans

# =========================
# Load models (cache)
# =========================
@st.cache_resource(show_spinner=False)
def load_abte(model_dir: str):
    # Ép dùng safetensors (an toàn với torch < 2.6)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir, use_safetensors=True)
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_sentiment(model_name: str):
    tok = AutoTok2.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
    mdl.eval()
    return tok, mdl

# =========================
# Inference
# =========================
def abte_predict_labels(words: List[str], tok, model, device: str = "cpu") -> List[str]:
    enc = tok(words, is_split_into_words=True, return_tensors="pt",
              truncation=True, padding=False)
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, L, C]
    pred_ids = logits.argmax(-1)[0].tolist()
    word_ids = enc.word_ids(0)

    labels_by_word = []
    seen = set()
    for i, wid in enumerate(word_ids):
        if wid is None:  # CLS/SEP
            continue
        if wid in seen:
            continue
        seen.add(wid)
        labels_by_word.append(ID2LABEL[pred_ids[i]])

    if len(labels_by_word) < len(words):
        labels_by_word += ["O"] * (len(words) - len(labels_by_word))
    return labels_by_word[:len(words)]

def classify_sentence(text: str, tok_s, mdl_s, device: str = "cpu") -> Tuple[str, list]:
    enc = tok_s(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl_s(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].tolist()
    idx = int(torch.tensor(probs).argmax().item())
    return SENT_LABELS[idx], probs

def classify_aspect(span_text: str, full_text: str, tok_s, mdl_s, device: str = "cpu", window: int = 40) -> Tuple[str, list]:
    try:
        start = full_text.lower().index(span_text.lower())
        left = max(0, start - window)
        right = min(len(full_text), start + len(span_text) + window)
        ctx = full_text[left:right]
    except ValueError:
        ctx = full_text
    return classify_sentence(ctx, tok_s, mdl_s, device=device)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ABTE Fine-tune Demo", page_icon="📝", layout="centered")
st.title("📝 ABTE Fine-tune Demo (Aspect Extraction + Sentiment)")

device = "cuda" if torch.cuda.is_available() else "cpu"
with st.sidebar:
    st.subheader("Model checkpoints")
    st.write("**Token classification (ABTE):**")
    st.code(ABTE_DIR or "(auto-detect failed)")
    st.write("**Sentence sentiment (3 lớp):**")
    st.code(SENT_MODEL_NAME)
    st.write(f"Device: **{device}**")

# Kiểm tra checkpoint ABTE
if ABTE_DIR is None:
    st.error(
        "Không tìm thấy checkpoint `.safetensors` cho ABTE.\n\n"
        "Hãy đảm bảo đã train với `save_safetensors=True`.\n"
        "Ưu tiên đường dẫn: `out_finetune/checkpoint-*/model.safetensors`."
    )
    st.stop()

# Load models
tok_abte, mdl_abte = load_abte(ABTE_DIR)
tok_s, mdl_s = load_sentiment(SENT_MODEL_NAME)
mdl_abte.to(device)
mdl_s.to(device)

default_text = "The chicken pho is amazing but the service is painfully slow."
text = st.text_area("Nhập câu tiếng Anh:", value=default_text, height=120)

if st.button("Phân tích"):
    if not text.strip():
        st.warning("Hãy nhập câu.")
    else:
        # 1) ABTE: token labels
        words = text.strip().split()
        labels = abte_predict_labels(words, tok_abte, mdl_abte, device=device)

        st.markdown("**Gán nhãn token (B/I/O):**")
        html = []
        for w, lb in zip(words, labels):
            bg = color_tag(lb)
            if bg:
                html.append(f"<span style='background:{bg}; padding:2px 4px; border-radius:4px;'>{w}</span>{label_badge(lb)}")
            else:
                html.append(f"{w}{label_badge(lb)}")
        st.markdown(" ".join(html), unsafe_allow_html=True)

        # 2) Aspect spans
        spans = extract_spans(words, labels)
        st.markdown("**Aspect terms:**")
        st.write(", ".join(f"`{s}`" for s in spans) if spans else "_Không có aspect term._")

        # 3) Sentence sentiment
        overall_label, _ = classify_sentence(text, tok_s, mdl_s, device=device)
        st.markdown(f"**Sentence Sentiment:** {overall_label}")

        # 4) Aspect polarity (ước lượng theo ngữ cảnh)
        st.markdown("**Aspect-level Polarity (ước lượng):**")
        if spans:
            rows = []
            for s in spans:
                lab, _ = classify_aspect(s, text, tok_s, mdl_s, device=device)
                rows.append({"aspect": s, "polarity": lab})
            st.table(rows)
        else:
            st.write("_Không có aspect để gán polarity._")

        with st.expander("Legend"):
            st.markdown("- 🟡 **B-Term**: bắt đầu một aspect term\n- 🩷 **I-Term**: phần tiếp theo của aspect term\n- ⬜ **O**: từ nền (không thuộc aspect)")
            st.caption("Polarity cho từng aspect dùng model sentence-level nên mang tính ước lượng nhanh cho demo.")
