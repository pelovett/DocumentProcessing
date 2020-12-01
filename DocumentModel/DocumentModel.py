import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from transformers import BertModel, BertConfig
from sklearn.metrics import f1_score


class DocumentModel(pl.LightningModule):
    acceptable_model_types = set(['first', 'sliding_window', 'aggregation'])

    def __init__(self,
                 num_classes: int,
                 transformer_base: str = 'bert-base-cased',
                 model_type: str = 'first'):
        super().__init__()
        assert model_type in DocumentModel.acceptable_model_types
        self.model_type = model_type
        self.base_config = BertConfig.from_pretrained(transformer_base)
        self.base_model = BertModel.from_pretrained(transformer_base)
        self.head = nn.Linear(self.base_config.hidden_size, num_classes)
        if model_type == 'aggregation':
            self.aggregator = None  # Figure out aggregation

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        x = batch['input_ids']
        if self.model_type == 'first':
            x_hat = self.head(self.base_model(x)[0][:, 0, :])
        elif self.model_type == 'sliding_window':
            cls_out = self.base_model(x)[0]
            window_map = batch['window_map']
            # Take the average of CLS representations between windows
            x_hat = torch.stack([
                cls_out[window, 0, :].mean(dim=0, keepdim=True)
                for window in window_map])
            x_hat = self.head(x_hat.squeeze(dim=1))
        x_hat = F.softmax(x_hat, dim=-1)
        return x_hat

    def training_step(self, batch, batch_idx):
        y = torch.stack(batch['label']).flatten()
        x_hat = self.forward(batch)
        loss = self.loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y = torch.stack(batch['label']).flatten()
        x_hat = self.forward(batch)
        loss = self.loss(x_hat, y)
        self.log('validation_loss', loss)
        pred = x_hat.argmax(dim=-1)
        return pred.item(), y.item()

    def validation_epoch_end(self, validation_step_outputs):
        guesses, correct_values = [], []
        total_samples = 0
        num_correct = 0
        for result in validation_step_outputs:
            total_samples += 1
            num_correct += (1 if result[0] == result[1] else 0)
            guesses.append(result[0])
            correct_values.append(result[1])
        self.log('validation_f1', f1_score(
            correct_values, guesses, average='macro'))
        self.log('validation_acc', num_correct/total_samples)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer
