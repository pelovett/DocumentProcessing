import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping, ModelCheckpoint

from WOSDataLoader import WOSDataModule
from WOSDataset import WOSDataset
from DocumentModel import DocumentModel


def main():
    mod_type = 'sliding_window'

    wos_data = WOSDataModule(batch_size=2, model_type=mod_type)
    model = DocumentModel(num_classes=7, model_type=mod_type)

    early_stop_callback = early_stopping.EarlyStopping(
        monitor='validation_f1',
        min_delta=0.001,
        patience=5,
        verbose=False,
        mode='max'
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor='validation_f1',
        dirpath='models/',
        filename='wos-bert-{epoch:02d}-{validation_f1:.2f}',
        mode='max'
    )

    trainer = pl.Trainer(callbacks=[early_stop_callback,
                                    model_checkpoint_callback],
                         log_every_n_steps=50)
    trainer.fit(model, wos_data)


if __name__ == '__main__':
    main()
