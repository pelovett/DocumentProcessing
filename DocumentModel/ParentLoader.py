import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import AutoTokenizer

from math import floor

from WOSDataset import WOSDataset


class ParentLoader(pl.LightningDataModule):
    acceptable_model_types = set(
        ['first', 'sliding_window', 'transformer', 'rnn'])
    transformer_sizes = {
        'bert-base-cased': 768,
        'allenai/longformer-base-4096': 768,
        'roberta-base': 768,
        'roberta-large': 1024
    }

    def __init__(self, config):
        super().__init__()

        assert config['model_type'] in ParentLoader.acceptable_model_types
        if config['transformer_name'] not in ParentLoader.transformer_sizes:
            print(
                f"Code not configured for model: {config['transformer_name']}")
            raise AttributeError
        self.window_size = config['window_size']
        self.data_dir = config['dataset_path']
        self.tokenizer_name = config['transformer_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.batch_size = config['batch_size']
        self.model_type = config['model_type']

    def prepare_data(self):
        raise NotImplementedError

    def collate_dict(self, batch):
        output_batch = dict()
        output_batch['text'] = [batch[i]['text']
                                for i in range(len(batch))]
        output_batch['label'] = [batch[i]['label']
                                 for i in range(len(batch))]
        if self.model_type == 'first':
            output_batch.update(self.tokenizer(
                output_batch['text'],
                padding=True,
                truncation=True,
                max_length=self.window_size,
                return_tensors='pt'))
        else:
            window_size = self.window_size - 2
            token_output = self.tokenizer(
                output_batch['text'],
                add_special_tokens=False,
                verbose=False)
            # window_key is an index of a batches subdivisions
            window_key = []
            window_index = 0
            output_batch['input_ids'] = []
            output_batch['attention_mask'] = []
            output_batch['token_type_ids'] = []
            for doc in token_output['input_ids']:
                temp_map = []
                doc_len = len(doc)
                prev_end = 0
                for j in range(min(doc_len // window_size, 8)):
                    # Group this current batch with its parent
                    temp_map.append(window_index)
                    window_index += 1
                    cur_end = (j + 1) * window_size
                    output_batch['input_ids'].append([101] +
                                                     doc[prev_end:cur_end] +
                                                     [102])
                    output_batch['token_type_ids'].append([0]*(window_size+2))
                    output_batch['attention_mask'].append([1]*(window_size+2))
                    prev_end = cur_end
                if doc_len % window_size != 0 and doc_len // window_size < 8:
                    temp_map.append(window_index)
                    window_index += 1
                    pad_value = window_size - len(doc[prev_end:])
                    # Adds the [CLS] and [SEP] tokens then pads to fill window
                    output_batch['input_ids'].append([101] +
                                                     doc[prev_end:] +
                                                     [102] +
                                                     [0] * pad_value)
                    output_batch['token_type_ids'].append([0]*(window_size+2))
                    attn_mask = [1] * (len(doc[prev_end:])+2) + \
                        [0] * pad_value
                    output_batch['attention_mask'].append(attn_mask)
                window_key.append(temp_map)
            if self.model_type in {'transformer', 'rnn'}:
                # Create attention mask
                # for some reason pytorch uses zeros as the unmasked value
                hidden_size = \
                    ParentLoader.transformer_sizes[self.tokenizer_name]
                temp = [torch.zeros((len(doc), hidden_size))
                        for doc in window_key]
                output_batch['doc_attention_mask'] = pad_sequence(
                    temp, batch_first=True, padding_value=1.0)
            output_batch['input_ids'] = torch.LongTensor(
                output_batch['input_ids'])
            output_batch['token_type_ids'] = torch.IntTensor(
                output_batch['token_type_ids'])
            output_batch['attention_mask'] = torch.IntTensor(
                output_batch['attention_mask'])
            output_batch['window_map'] = window_key
        return output_batch

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
