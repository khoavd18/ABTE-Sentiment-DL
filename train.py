# train.py
import os
from transformers import TrainingArguments, Trainer
from init import build_experiment, compute_metrics_token_f1

# ======= đổi mỗi dòng này để chọn mô hình =======
# conv1d | lstm | transformer | finetune
model, preprocessed_ds, extras = build_experiment(model_name="finetune")
# ================================================

training_args = TrainingArguments(
    output_dir=f"out_{extras.get('run_name','run')}",
    logging_dir="logs",
    learning_rate=2e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=100,
    weight_decay=0.01,
    eval_strategy="epoch", 
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=2, 
    load_best_model_at_end=True,
    metric_for_best_model="f1-score",
    report_to="none",
    fp16=True,
    save_safetensors=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_ds["train"],
    eval_dataset=preprocessed_ds["test"],
    compute_metrics=compute_metrics_token_f1,
    tokenizer=extras.get("tokenizer"),       
    data_collator=extras.get("data_collator"),
)
trainer.train()