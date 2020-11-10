import torch.nn as nn
import torch
import pytorch_lightning as pl
from transformers import BertModel, BertConfig


class DocumentModel(pl.LightningModule):

    def __init__(self, num_classes: int, transformer_base: str = 'bert-base-cased'):
        super().__init__()
        self.base_config = BertConfig.from_pretrained(transformer_base)
        self.base_model = BertModel.from_pretrained(transformer_base)
        self.head = nn.Linear(self.base_config.hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(torch.Tensor([[1]]))

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['label'].flatten()
        x_hat = self.head(self.base_model(x)[1])
        x_hat = nn.functional.softmax(x_hat, dim=-1)
        loss = self.loss(x_hat, y.to(torch.long))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
