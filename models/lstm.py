# models/abte_lstm.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class ABTELSTMClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes,
                 embedding_dim=256, hidden_dim=256, pad_idx=0):

        super().__init__(config)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        logits = self.fc(outputs)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1), labels)

        return {"loss": loss, "logits": logits}
