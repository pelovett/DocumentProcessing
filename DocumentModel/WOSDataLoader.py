import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
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

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.eval_split, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size)
