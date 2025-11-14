# models/abte_transformer.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class ABTETransformerClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes,
                 max_len=512, embedding_dim=256, num_heads=8,
                 num_layers=6, hidden_dim=1024, pad_idx=0):

        super().__init__(config)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx = pad_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        batch_size, seq_length = input_ids.size()

        positions = torch.arange(
            seq_length, device=input_ids.device).unsqueeze(0)

        outputs = self.embedding(input_ids) + self.position_embedding(positions)

        outputs = outputs.permute(1, 0, 2)
        outputs = self.transformer_encoder(outputs)

        outputs = outputs.permute(1, 0, 2) #=> BxSxE
        logits = self.fc(outputs)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1), labels)
        return {"loss": loss, "logits": logits}
