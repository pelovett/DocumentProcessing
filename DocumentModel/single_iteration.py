import pytorch_lightning as pl
import yaml
from math import floor

from ImdbDataLoader import ImdbDataModule
from WOSDataLoader import WOSDataModule
from HypeDataLoader import HypeDataModule
from DocumentModel import DocumentModel


def main():
    config = yaml.load(open('conf/default.yml', 'r'), Loader=yaml.SafeLoader)
    if config['dataset'] == 'hyperpartisan_news':
        data_loader = HypeDataModule(config)
    elif config['dataset'] == 'web_of_science':
        data_loader = WOSDataModule(config)
    elif config['dataset'] == 'Imdb':
        data_loader = ImdbDataModule(config)
    # model = DocumentModel(num_classes=7, model_type=mod_type)

    data_loader.prepare_data()
    train_iter = data_loader.train_dataloader()
    train_iter = iter(train_iter)
    batch_sizes = []
    for batch in train_iter:
        batch_sizes.append(batch['input_ids'].shape[0])
    batch_sizes = sorted(batch_sizes)
    print(f'Average batch size: {sum(batch_sizes)/len(batch_sizes)}')
    print(f'Median batch size: {batch_sizes[floor(len(batch_sizes)/2)]}')
    print(f'Max batch size: {max(batch_sizes)}')
    # print(model.training_step(batch, 0))
    # print(model.validation_step(batch, 0))


if __name__ == '__main__':
    main()
