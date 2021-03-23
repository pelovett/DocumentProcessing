from DocumentModel import DocumentModel
from PatentDataLoader import PatentDataModule
from WOSDataLoader import WOSDataModule
from HypeDataLoader import HypeDataModule
from ImdbDataLoader import ImdbDataModule
from ArxivDataLoader import ArxivDataModule

import yaml
import sys
import argparse
from time import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping, ModelCheckpoint, Callback


def main(config, gpus, checkpoint):
    patience = config['early_stop_patience']
    min_delta = config['early_stop_delta']
    log_dir = config['logging_dir']
    run_name = config['experiment_name']
    seed = config['seed']

    if config['dataset'] in {'small_hyper', 'hyperpartisan_news'}:
        data_module = HypeDataModule(config)
    elif config['dataset'] == 'web_of_science':
        data_module = WOSDataModule(config)
    elif config['dataset'] == 'imdb':
        data_module = ImdbDataModule(config)
    elif config['dataset'] == 'arxiv':
        data_module = ArxivDataModule(config)
    elif config['dataset'] == 'patent':
        data_module = PatentDataModule(config)
    else:
        print(f"### Unknown dataset: {config['dataset']} ###")
        raise NotImplementedError

    model = DocumentModel(config)

    logger = pl.loggers.TensorBoardLogger(log_dir,
                                          default_hp_metric=True,
                                          name=run_name)
    logger.log_hyperparams(config)

    class TimeTestEpochCallback(Callback):
        start_time = 0

        def on_test_epoch_start(self, trainer, pl_module):
            self.start_time = time()

        def on_test_epoch_end(self, trainer, pl_module):
            end_time = time()
            print(f'Test epoch wall clock time: {end_time - self.start_time}')

    trainer = pl.Trainer(callbacks=[TimeTestEpochCallback()],
                         logger=logger,
                         gpus=[gpus] if gpus != None else None)
    pl.trainer.seed_everything(seed)

    data_module.prepare_data()
    trainer.test(model,
                 test_dataloaders=data_module.test_dataloader(),
                 ckpt_path=checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default=None)
    parser.add_argument('--gpus', default=None, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    args = parser.parse_args()

    try:
        config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
        main(config, args.gpus, args.checkpoint)
    except IndexError:
        print(f'Must pass in config file. Usage: python3 train_script.py config_file')
    except FileNotFoundError:
        print(f'Unable to find file {sys.argv[1]}')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
