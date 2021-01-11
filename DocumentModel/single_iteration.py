import pytorch_lightning as pl
import yaml

from WOSDataLoader import WOSDataModule
from HypeDataLoader import HypeDataModule
from DocumentModel import DocumentModel


def main():
    config = yaml.load(open('conf/default.yml', 'r'), Loader=yaml.SafeLoader)
    data_loader = HypeDataModule(config)
    #model = DocumentModel(num_classes=7, model_type=mod_type)

    data_loader.prepare_data()
    train_iter = data_loader.train_dataloader()
    train_iter = iter(train_iter)
    batch = next(train_iter)
    #print(model.training_step(batch, 0))
    #print(model.validation_step(batch, 0))


if __name__ == '__main__':
    main()
