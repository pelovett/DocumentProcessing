from torch.utils.data import random_split, DataLoader

from math import floor

from ParentLoader import ParentLoader
from WOSDataset import WOSDataset


class WOSDataModule(ParentLoader):

    def __init__(self, config):
        super().__init__(config)

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
        return DataLoader(self.train_split,
                          collate_fn=self.collate_dict,
                          batch_size=self.batch_size,
                          num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.eval_split,
                          collate_fn=self.collate_dict,
                          batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_split,
                          collate_fn=self.collate_dict,
                          batch_size=1)
