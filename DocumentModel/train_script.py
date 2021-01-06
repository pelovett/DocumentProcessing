import sys
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping, ModelCheckpoint
import yaml

from WOSDataLoader import WOSDataModule
from WOSDataset import WOSDataset
from DocumentModel import DocumentModel


def main(config, gpus):
    patience = config['early_stop_patience']
    min_delta = config['early_stop_delta']
    log_dir = config['logging_dir']
    run_name = config['experiment_name']
    seed = config['seed']

    wos_data = WOSDataModule(config)
    model = DocumentModel(config)

    early_stop_callback = early_stopping.EarlyStopping(
        monitor='validation_f1',
        min_delta=min_delta,
        patience=patience,
        verbose=False,
        mode='max'
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor='validation_f1',
        dirpath='models/',
        filename='run_name',
        mode='max'
    )

    logger = pl.loggers.TensorBoardLogger(log_dir,
                                          default_hp_metric=True,
                                          name=run_name)
    logger.log_hyperparams(config)

    trainer = pl.Trainer(callbacks=[early_stop_callback,
                                    model_checkpoint_callback],
                         logger=logger,
                         log_every_n_steps=50,
                         progress_bar_refresh_rate=0,
                         gpus=[gpus] if gpus != None else None)
    # precision=16)
    pl.trainer.seed_everything(seed)
    trainer.fit(model, wos_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default=None)
    parser.add_argument('--gpus', default=None, type=int)
    args = parser.parse_args()

    try:
        config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
        main(config, args.gpus)
    except IndexError:
        print(f'Must pass in config file. Usage: python3 train_script.py config_file')
    except FileNotFoundError:
        print(f'Unable to find file {sys.argv[1]}')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
