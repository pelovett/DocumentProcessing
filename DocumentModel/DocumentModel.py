import torch.nn as nn
import torch.nn.functional as F
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
        if x.shape[-1] > 512:
            x = x[:, :512]
        x_hat = self.head(self.base_model(x)[0][:, 0, :])
        x_hat = F.softmax(x_hat, dim=-1)
        loss = self.loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['label'].flatten()  # Fix this for batch_num > 1
        if x.shape[-1] > 512:
            x = x[:, :512]
        x_hat = self.head(self.base_model(x)[0][:, 0, :])
        x_hat = F.softmax(x_hat, dim=-1)
        loss = self.loss(x_hat, y)
        self.log('validation_loss', loss)
        pred = x_hat.argmax(dim=-1)
        return pred == y

    def validation_epoch_end(self, validation_step_outputs):
        total_samples = 0
        num_correct = 0
        for result in validation_step_outputs:
            total_samples += result.shape[0]  # Fix this for batch_num > 1
            num_correct += result.int().sum()
        self.log('validation_acc', num_correct/total_samples*100)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)
        return optimizer
