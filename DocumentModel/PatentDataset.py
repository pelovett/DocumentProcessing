from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import logging
import json


class PatentDataset(Dataset):
    logger = logging.getLogger(__name__)

    def __init__(self,
                 file_path: str = './data/patent/',
                 tokenizer: str = 'bert-base-cased',
                 split: str = 'Train'):
        super().__init__()
        self.file_path = file_path
        assert split in {'Train', 'Validation', 'Test'}
        self.x = []
        self.y = []
        self.label_counts = dict()

        with open(file_path+'USPTO-labels.json', 'r') as in_file:
            tag_map = json.loads(in_file.read())

        try:
            if split == 'Train':
                file_name = 'USPTO-train.json'
            elif split == 'Validation':
                file_name = 'USPTO-validation.json'
            elif split == 'Test':
                file_name = 'USPTO-test.json'

            with open(file_path+file_name, 'r') as in_file:
                for line in in_file:
                    cur = json.loads(line)
                    self.x.append(cur['Title']+' '+cur['Abstract'])
                    targets = []
                    for sub in cur['Subclass_labels']:
                        label_index = tag_map[sub]
                        targets.append(label_index)
                        if label_index in self.label_counts:
                            self.label_counts[label_index] += 1
                        else:
                            self.label_counts[label_index] = 1
                    self.y.append(targets)
            self.num_labels = len(tag_map)

        except FileNotFoundError:
            logging.error(
                f'### Failed to find data file: {self.file_path}X.txt')
            raise FileNotFoundError

        try:
            assert len(self.x) == len(self.y)
        except AssertionError:
            print(len(self.x), len(self.y))
            raise AssertionError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {}
        sample['text'] = self.x[idx]
        sample['label'] = torch.zeros(self.num_labels)
        for lab in self.y[idx]:
            sample['label'][lab] = 1
        sample['raw_label'] = torch.LongTensor(self.y[idx])
        return sample

    def __iter__(self):
        return iter(range(self.__len__()))
