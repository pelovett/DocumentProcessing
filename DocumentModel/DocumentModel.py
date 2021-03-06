import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch
import pytorch_lightning as pl
import numpy as np
from transformers import AutoModel, AutoConfig
from sklearn.metrics import f1_score


class DocumentModel(pl.LightningModule):
    acceptable_model_types = set(
        ['first', 'sliding_window', 'transformer', 'rnn'])
    # warm up lr (200 value comes from longformer paper)
    # RoBERTa uses warmup on first 6% of steps followed
    NUM_WARMUP_STEPS = 200

    def __init__(self, config):
        super().__init__()
        self.model_type = config['model_type']
        self.learning_rate = config['learning_rate']
        dropout = config['dropout']
        transformer_base = config['transformer_name']
        num_classes = config['num_classes']
        self.multilabel = True if config['dataset'] == 'patent' else False
        self.config = config

        assert self.model_type in DocumentModel.acceptable_model_types
        self.base_config = AutoConfig.from_pretrained(transformer_base)
        self.base_model = AutoModel.from_pretrained(transformer_base)
        # gradient_checkpointing=True)
        self.head = nn.Linear(self.base_config.hidden_size, num_classes)
        if self.model_type == 'transformer':
            self.aggregator = nn.TransformerEncoderLayer(
                self.base_config.hidden_size,
                nhead=4,
                dim_feedforward=config['hidden_size'],
                dropout=dropout)
        elif self.model_type == 'rnn':
            self.aggregator = nn.GRU(
                hidden_size=config['hidden_size'],
                input_size=self.base_config.hidden_size,
                batch_first=True,
                dropout=dropout)
            self.head = nn.Linear(config['hidden_size'], num_classes)
        if self.multilabel:
            label_weights = [config['counts'][i]
                             for i in range(num_classes)]
            total_num_samples = sum(label_weights)
            label_weights = [total_num_samples / x for x in label_weights]
            self.loss = nn.BCEWithLogitsLoss(
                weight=torch.FloatTensor(label_weights))
        else:
            self.loss = nn.CrossEntropyLoss()
        self.f1_score = pl.metrics.classification.F1(num_classes=num_classes)
        self.accuracy = pl.metrics.classification.Accuracy()

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
        elif self.model_type == 'transformer':
            cls_out = self.base_model(x)[0]
            window_map = batch['window_map']
            x_hat = pad_sequence([cls_out[doc, 0, :] for doc in window_map],
                                 batch_first=True)
            # not sure why there's no batch first option
            x_hat = self.aggregator(x_hat.transpose(1, 0)).transpose(1, 0)
            # multiply by mask to avoid impact of pad
            x_hat = torch.mean(x_hat, dim=1)
            x_hat = self.head(x_hat)
        elif self.model_type == 'rnn':
            cls_out = self.base_model(x)[0]
            window_map = batch['window_map']
            x_hat = pad_sequence([cls_out[doc, 0, :] for doc in window_map],
                                 batch_first=True)
            x_hat = self.aggregator(x_hat)[0]
            x_hat = torch.mean(x_hat, dim=1)
            x_hat = self.head(x_hat)
        x_hat = F.softmax(x_hat, dim=-1)
        return x_hat

    def training_step(self, batch, batch_idx):
        if self.multilabel:
            y = torch.stack(batch['label'])
        else:
            y = torch.stack(batch['label']).flatten()
        x_hat = self.forward(batch)
        loss = self.loss(x_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.multilabel:
            y = torch.stack(batch['label'])
        else:
            y = torch.stack(batch['label']).flatten()
        x_hat = self.forward(batch)
        loss = self.loss(x_hat, y)
        self.log('validation_loss', loss)
        if not self.multilabel:
            pred = x_hat.argmax(dim=-1)
        else:
            pred = x_hat
        return pred.detach(), y.detach()

    def validation_epoch_end(self, validation_step_outputs):
        guesses = torch.cat([x[0] for x in validation_step_outputs])
        labels = torch.cat([x[1] for x in validation_step_outputs])
        val_f1 = self.f1_score(guesses, labels)
        val_acc = self.accuracy(guesses, labels)
        self.log('validation_f1', val_f1)
        self.log('validation_acc', val_acc)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, validation_step_outputs):
        guesses = torch.cat([x[0] for x in validation_step_outputs])
        labels = torch.cat([x[1] for x in validation_step_outputs])
        test_f1 = self.f1_score(guesses, labels)
        test_acc = self.accuracy(guesses, labels)
        self.log('test_f1', test_f1)
        self.log('test_acc', test_acc)

    def optimizer_step(self, current_epoch, batch_nb, optimizer,
                       optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # accumulation means we only backpropagate after backpropagating
        # on a set number of batches
        if self.config['accumulate_num'] > 1:
            if self.trainer.global_step % self.config['accumulate_num'] != 0:
                return

        # learning rate warm-up
        # flag check is verbose for readability and back compatibility
        if 'lr_warmup' in self.config and self.config['lr_warmup'] == True:
            warmup_steps = DocumentModel.NUM_WARMUP_STEPS
            if self.trainer.global_step < warmup_steps:
                lr_scale = min(1.,
                               float(self.trainer.global_step + 1) /
                               float(warmup_steps))
            else:
                total_num_steps = len(self.trainer.train_dataloader) * \
                    self.config['maximum_num_epochs'] - warmup_steps
                adjusted_step_count = self.trainer.global_step - warmup_steps
                lr_scale = min(1.,
                               1 - (adjusted_step_count / total_num_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.config['learning_rate']

        optimizer.step(closure=closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
