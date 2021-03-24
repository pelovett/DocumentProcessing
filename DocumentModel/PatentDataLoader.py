from torch.utils.data import random_split, DataLoader

from math import floor
import json

from ParentLoader import ParentLoader
from PatentDataset import PatentDataset


class PatentDataModule(ParentLoader):

    def __init__(self, config):
        super().__init__(config)
        with open(self.data_dir+'USPTO-labels.json', 'r') as in_file:
            tag_map = json.loads(in_file.read())

        self.label_counts = dict()
        with open(self.data_dir+'USPTO-train.json', 'r') as in_file:
            for line in in_file:
                cur = json.loads(line)
                for sub in cur['Subclass_labels']:
                    label_index = tag_map[sub]
                    if label_index in self.label_counts:
                        self.label_counts[label_index] += 1
                    else:
                        self.label_counts[label_index] = 1

    def prepare_data(self):
        self.train_dataset = PatentDataset(self.data_dir,
                                           self.tokenizer_name,
                                           'Train')

        self.val_dataset = PatentDataset(self.data_dir,
                                         self.tokenizer_name,
                                         'Validation')

        self.test_dataset = PatentDataset(self.data_dir,
                                          self.tokenizer_name,
                                          'Test')

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
        return DataLoader(self.test_dataset,
                          collate_fn=self.collate_dict,
                          batch_size=self.batch_size,
                          num_workers=1)
