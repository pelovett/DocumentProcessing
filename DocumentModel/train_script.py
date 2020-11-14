import pytorch_lightning as pl

from WOSDataLoader import WOSDataModule
from WOSDataset import WOSDataset
from DocumentModel import DocumentModel


def main():
    wos_data = WOSDataModule()
    model = DocumentModel(num_classes=7)

    trainer = pl.Trainer(log_every_n_steps=50)
    trainer.fit(model, wos_data)


if __name__ == '__main__':
    main()
