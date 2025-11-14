# abte/__init__.py
import numpy as np
from datasets import load_dataset
from seqeval.metrics import f1_score

# word-level tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# hf finetune
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, PretrainedConfig

# models tự viết
from models.cnn import ABTEConv1DClassifier
from models.lstm import ABTELSTMClassifier
from models.transformers import ABTETransformerClassifier

ID2LABEL = {0: "O", 1: "B-Term", 2: "I-Term"}
LABEL2ID = {"O": 0, "B-Term": 1, "I-Term": 2}

# ============== metric chung ==============
def compute_metrics_token_f1(eval_pred):
    preds, labs = eval_pred
    preds = np.argmax(preds, axis=-1)
    true_preds, true_labels = [], []
    for p, l in zip(preds, labs):
        ps, ls = [], []
        for pi, li in zip(p, l):
            if li == -100:  # bỏ pad
                continue
            ps.append(ID2LABEL[int(pi)]); ls.append(ID2LABEL[int(li)])
        true_preds.append(ps); true_labels.append(ls)
    return {"f1-score": f1_score(true_labels, true_preds)}

# ============== helpers word-level (custom) ==============
def _build_wordlevel_tokenizer(corpus, vocab_size=5000):
    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainer=trainers.WordLevelTrainer(
        vocab_size=vocab_size, special_tokens=["<pad>", "<unk>"]
    ))
    return tk

def _encode_wordlevel(ds, tk: Tokenizer):
    max_len = max(len(x) for x in ds["train"]["Tokens"])
    pad_id = tk.token_to_id("<pad>") or 0

    def pad_trunc(seq, pad_val):
        return seq + [pad_val]*(max_len-len(seq)) if len(seq) < max_len else seq[:max_len]

    def mapper(batch):
        outs = {"input_ids": [], "labels": []}
        for tokens, tags in zip(batch["Tokens"], batch["Tags"]):
            ids  = [tk.token_to_id(t.lower()) or 0 for t in tokens]
            labs = [int(x) for x in tags]
            outs["input_ids"].append(pad_trunc(ids, pad_id))
            outs["labels"].append(pad_trunc(labs, -100))
        return outs

    enc = ds.map(mapper, batched=True)
    return enc, pad_id, max_len

# ============== helpers finetune (subword) ==============
def _encode_finetune_subword(ds, tok):
    def mapper(batch):
        outs = {"input_ids": [], "labels": []}
        for tokens, tags in zip(batch["Tokens"], batch["Tags"]):
            subs_all, labs_all = [], []
            for t, g in zip(tokens, tags):
                subs = tok.tokenize(t)
                subs_all += subs
                labs_all += [int(g)] * len(subs)
            outs["input_ids"].append(tok.convert_tokens_to_ids(subs_all))
            outs["labels"].append(labs_all)
        return outs

    enc = ds.map(
        mapper, batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in ["Tokens","Tags"]]
    )
    return enc

# ============== factory duy nhất bạn gọi từ train.py ==============
def build_experiment(model_name="conv1d", hf_dataset="thainq107/abte-restaurants",
                     pretrained="distilbert-base-uncased", vocab_size=5000):
    ds = load_dataset(hf_dataset)

    if model_name == "finetune":
        tok = AutoTokenizer.from_pretrained(pretrained)
        enc = _encode_finetune_subword(ds, tok)
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
        )
        extras = {
            "tokenizer": tok,
            "data_collator": DataCollatorForTokenClassification(tokenizer=tok),
            "run_name": f"finetune-{pretrained.rsplit('/',1)[-1]}"
        }
        return model, enc, extras

    # 3 model custom (word-level)
    corpus = [" ".join(t) for t in ds["train"]["Tokens"]]
    tk = _build_wordlevel_tokenizer(corpus, vocab_size=vocab_size)
    enc, pad_id, max_len = _encode_wordlevel(ds, tk)
    vocab = len(tk.get_vocab())
    
    config = PretrainedConfig()

    if model_name == "conv1d":
        model = ABTEConv1DClassifier(config, vocab, num_classes=3)
    elif model_name == "lstm":
        model = ABTELSTMClassifier(config, vocab, num_classes=3)
    else:
        model = ABTETransformerClassifier(config, vocab, num_classes=3)

    extras = {"run_name": model_name}  # custom không cần tokenizer/collator
    return model, enc, extras
