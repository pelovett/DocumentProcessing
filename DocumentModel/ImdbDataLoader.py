from torch.utils.data import random_split, DataLoader

from math import floor

from ParentLoader import ParentLoader
from ImdbDataset import ImdbDataset


class ImdbDataModule(ParentLoader):

    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self):
        self.train_dataset = ImdbDataset(self.data_dir,
                                         self.tokenizer_name,
                                         'train/')
        self.test_dataset = ImdbDataset(self.data_dir,
                                        self.tokenizer_name,
                                        'test/')

        total_length = len(self.train_dataset)
        train_length = floor(total_length * 0.7)
        eval_length = total_length - train_length

        self.train_split, self.eval_split = \
            random_split(self.train_dataset,
                         (train_length,
                          eval_length)
                         )

    def train_dataloader(self):
        return DataLoader(self.train_split,
                          collate_fn=self.collate_dict,
                          batch_size=self.batch_size,
                          num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.eval_split,
                          collate_fn=self.collate_dict,
                          batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          collate_fn=self.collate_dict,
                          batch_size=1)
