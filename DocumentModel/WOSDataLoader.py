import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from math import floor

from WOSDataset import WOSDataset


class WOSDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str = './data/WOS11967/',
                 tokenizer_name: str = 'bert-base-cased',
                 batch_size: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size

    def prepare_data(self):
        self.dataset = WOSDataset(self.data_dir, self.tokenizer_name)

        total_length = len(self.dataset)
        train_length = floor(total_length * 0.7)
        eval_length = total_length - train_length
        test_length = floor(eval_length * 1/3)
        eval_length = eval_length - test_length

        self.train_split, self.eval_split, self.test_split = \
            random_split(self.dataset,
                         (train_length,
                          eval_length,
                          test_length)
                         )

    def collate_dict(self, batch):
        output_batch = dict()
        if len(batch) > 1:
            output_batch['input_ids'] = pad_sequence(
                        [batch[i]['input_ids'] for i in range(len(batch))],
                batch_first=True
            )
            output_batch['label'] = pad_sequence(
                [batch[i]['label'] for i in range(len(batch))],
                batch_first=True
            )
        else:
            output_batch['input_ids'] = batch[0]['input_ids'].unsqueeze(dim=0)
            output_batch['label'] = batch[0]['label'].unsqueeze(dim=0)
        output_batch['mask'] = (output_batch['label'] != 0)
        return output_batch

    def train_dataloader(self):
        return DataLoader(self.train_split,
                          collate_fn=self.collate_dict,
                          batch_size=self.batch_size,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.eval_split,
                          collate_fn=self.collate_dict,
                          batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_split,
                          collate_fn=self.collate_dict,
                          batch_size=1)
