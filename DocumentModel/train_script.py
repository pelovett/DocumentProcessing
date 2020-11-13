import pytorch_lightning as pl

from DocumentModel.WOSDataLoader import WOSDataModule
from DocumentModel.WOSDataset import WOSDataset
from DocumentModel import DocumentModel


def main():
    wos_data = WOSDataModule()
    model = DocumentModel(num_classes=7)

    trainer = pl.Trainer()
    trainer.fit(model, wos_data)


if __name__ == '__main__':
    main()
