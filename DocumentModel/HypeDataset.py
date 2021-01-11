from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import logging


class HypeDataset(Dataset):
    logger = logging.getLogger(__name__)

    def __init__(self,
                 file_path: str = './data/hyperpartisan_news/',
                 tokenizer: str = 'bert-base-cased',
                 split: str = 'Train'):
        super().__init__()
        self.file_path = file_path
        assert split in {'Train', 'Validation'}
        self.x = []
        self.y = []

        try:
            if split == 'Train':
                dir_name = 'training/'
            elif split == 'Validation':
                dir_name = 'validation/'

            with open(file_path+dir_name+'Y.txt', 'r') as in_file:
                for line in in_file:
                    cur = line.strip().split('\t')  # TODO change for other labels
                    self.y.append(1 if cur[1] == 'true' else 0)

            tag_set = set()
            with open(file_path+dir_name+'X.txt', 'r') as in_file:
                for line in in_file:
                    self.x.append(line.strip())

        except FileNotFoundError:
            logging.error(
                f'### Failed to find data file: {self.file_path}X.txt')
            raise FileNotFoundError
        self.num_labels = 2  # TODO change for other labels
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
        sample['label'] = torch.LongTensor([int(self.y[idx])])
        return sample

    def __iter__(self):
        return iter(range(self.__len__()))
