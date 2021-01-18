from torch.utils.data import random_split, DataLoader

from math import floor

from ParentLoader import ParentLoader
from ArxivDataset import ArxivDataset


class ArxivDataModule(ParentLoader):

    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self):
        self.train_dataset = ArxivDataset(self.data_dir,
                                          self.tokenizer_name,
                                          'train/')
        self.val_dataset = ArxivDataset(self.data_dir,
                                        self.tokenizer_name,
                                        'validation/')
        self.test_dataset = ArxivDataset(self.data_dir,
                                         self.tokenizer_name,
                                         'test/')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self.collate_dict,
                          batch_size=self.batch_size,
                          num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          collate_fn=self.collate_dict,
                          batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          collate_fn=self.collate_dict,
                          batch_size=1)
