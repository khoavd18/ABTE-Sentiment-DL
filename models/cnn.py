# models/abte_cnn.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class ABTEConv1DClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes, embedding_dim=256, num_filters=256, kernel_size=3, pad_idx=0):
        super().__init__(config)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.fc = nn.Linear(num_filters, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)

        # conv1d
        features = torch.relu(self.conv(embedded))

        features = features.permute(0, 2, 1)
        logits = self.fc(features)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1), labels)
        return {"loss": loss, "logits": logits}
