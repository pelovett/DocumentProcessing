from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import logging


class WOSDataset(Dataset):
    logger = logging.getLogger(__name__)

    def __init__(self, file_path: str = './data/WOS11967/', tokenizer: str = 'bert-base-cased'):
        super().__init__()
        self.file_path = file_path

        self.x = []
        self.y = []

        try:
            with open(self.file_path+'X.txt', 'r') as feature_data_file:
                for line in feature_data_file:
                    self.x.append(line.strip())
        except FileNotFoundError:
            logging.error(
                f'### Failed to find data file: {self.file_path}X.txt')
            raise FileNotFoundError

        tag_set = set()
        try:
            with open(self.file_path+'YL1.txt', 'r') as feature_data_file:
                for line in feature_data_file:
                    self.y.append(int(line.strip()))
                    tag_set.add(int(line.strip()))
        except FileNotFoundError:
            logging.error(
                f'### Failed to find data file: {self.file_path}YL1.txt')
            raise FileNotFoundError
        self.num_labels = len(tag_set)
        assert len(self.x) == len(self.y)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {}
        sample['features'] = self.tokenizer.tokenize(self.x[idx])
        sample['input_ids'] = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(sample['features'])
        )
        sample['label'] = torch.Tensor([int(self.y[idx])])
        return sample

    def __iter__(self):
        return iter(range(self.__len__()))
