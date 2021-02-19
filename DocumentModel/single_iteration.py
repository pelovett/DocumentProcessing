import pytorch_lightning as pl
import yaml
from math import floor

from ArxivDataLoader import ArxivDataModule
from ImdbDataLoader import ImdbDataModule
from WOSDataLoader import WOSDataModule
from HypeDataLoader import HypeDataModule
from PatentDataLoader import PatentDataModule
from DocumentModel import DocumentModel


def main():
    config = yaml.load(open('conf/default.yml', 'r'), Loader=yaml.SafeLoader)
    if config['dataset'] == 'hyperpartisan_news':
        data_loader = HypeDataModule(config)
    elif config['dataset'] == 'web_of_science':
        data_loader = WOSDataModule(config)
    elif config['dataset'] == 'Imdb':
        data_loader = ImdbDataModule(config)
    elif config['dataset'] == 'arxiv':
        data_loader = ArxivDataModule(config)
    elif config['dataset'] == 'patent':
        data_loader = PatentDataModule(config)

    model = DocumentModel(config)

    data_loader.prepare_data()
    train_iter = data_loader.train_dataloader()
    train_iter = iter(train_iter)
    next(train_iter)
    batch = next(train_iter)
    print(batch['text'])
    print(batch['label'])
    print(model.training_step(batch, 0))
    print(model.validation_step(batch, 0))


if __name__ == '__main__':
    main()
