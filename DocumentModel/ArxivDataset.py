from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import logging


label_dict = {
    'cs_AI': 0, 'cs_CV': 1, 'cs_IT': 2, 'cs_PL': 3, 'math_AC': 4,
    'math_ST': 5, 'cs_CE': 6, 'cs_DS': 7, 'cs_NE': 8, 'cs_SY': 9,
    'math_GR': 10
}


class ArxivDataset(Dataset):
    logger = logging.getLogger(__name__)

    def __init__(self,
                 file_path: str = './data/arxiv/',
                 tokenizer: str = 'bert-base-cased',
                 split: str = 'train/'):
        super().__init__()
        self.file_path = file_path
        assert split in {'train/', 'validation/', 'test/'}
        self.x = []
        self.y = []

        try:
            for subdir in listdir(file_path+split):
                label = label_dict[subdir]
                dir_path = file_path+split+subdir+'/'
                for review in listdir(dir_path):
                    with open(dir_path+review, 'r') as in_file:
                        self.x.append(in_file.read().strip())
                    self.y.append(label)
        except FileNotFoundError:
            logging.error(
                f'### Failed to find data file: {dir_path+review}')
            raise FileNotFoundError
        self.num_labels = len(label_dict)
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
