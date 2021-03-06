from torch.utils.data import random_split, DataLoader

from math import floor

from ParentLoader import ParentLoader
from HypeDataset import HypeDataset


class HypeDataModule(ParentLoader):

    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self):
        self.train_dataset = HypeDataset(self.data_dir,
                                         self.tokenizer_name,
                                         'Train')
        self.val_dataset = HypeDataset(self.data_dir,
                                       self.tokenizer_name,
                                       'Validation')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self.collate_dict,
                          batch_size=self.batch_size,
                          num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          collate_fn=self.collate_dict,
                          batch_size=self.batch_size,
                          num_workers=1)

    def test_dataloader(self):
        raise NotImplementedError
