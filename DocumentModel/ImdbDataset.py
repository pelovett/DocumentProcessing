from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import logging


class ImdbDataset(Dataset):
    logger = logging.getLogger(__name__)

    def __init__(self,
                 file_path: str = './data/aclImdb/',
                 tokenizer: str = 'bert-base-cased',
                 split: str = 'train/'):
        super().__init__()
        self.file_path = file_path
        assert split in {'train/', 'test/'}
        self.x = []
        self.y = []

        try:
            review = ''
            for subdir, label in [('neg/', 0), ('pos/', 1)]:
                dir_path = file_path+split+subdir
                for review in listdir(dir_path):
                    with open(dir_path+review, 'r') as in_file:
                        self.x.append(in_file.read().strip())
                    self.y.append(label)

        except FileNotFoundError:
            logging.error(
                f'### Failed to find data file: {dir_path+review}')
            raise FileNotFoundError
        self.num_labels = 2
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
