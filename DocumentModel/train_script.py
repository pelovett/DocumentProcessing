import sys
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping, ModelCheckpoint
import yaml

from ArxivDataLoader import ArxivDataModule
from ImdbDataLoader import ImdbDataModule
from HypeDataLoader import HypeDataModule
from WOSDataLoader import WOSDataModule
from PatentDataLoader import PatentDataModule
from DocumentModel import DocumentModel


def main(config, gpus):
    patience = config['early_stop_patience']
    max_epochs = config['maximum_num_epochs']
    min_delta = config['early_stop_delta']
    log_dir = config['logging_dir']
    run_name = config['experiment_name']
    seed = config['seed']

    if config['batch_size'] > 8:
        assert config['batch_size'] in {16, 32}
        if config['dataset'] in {'small_hyper', 'hyperpartisan_news'}:
            if config['batch_size'] == 16:
                accumulate_num = 4
            else:
                accumulate_num = 8
            config['batch_size'] = 4
        else:
            if config['batch_size'] == 16:
                accumulate_num = 2
            else:
                accumulate_num = 4
            config['batch_size'] = 8
    else:
        accumulate_num = 1

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
        # Make counts available for weighting loss function
        config['counts'] = data_module.train_dataloader.label_counts
    else:
        print(f"### Unknown dataset: {config['dataset']} ###")
        raise NotImplementedError
    pl.trainer.seed_everything(seed)
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
        filename=run_name,
        mode='max'
    )

    logger = pl.loggers.TensorBoardLogger(log_dir,
                                          default_hp_metric=True,
                                          name=run_name)
    logger.log_hyperparams(config)

    trainer = pl.Trainer(callbacks=[early_stop_callback,
                                    model_checkpoint_callback],
                         logger=logger,
                         enable_pl_optimizer=True,
                         gradient_clip_val=0.25,
                         accumulate_grad_batches=accumulate_num,
                         max_epochs=max_epochs,
                         log_every_n_steps=50,
                         progress_bar_refresh_rate=0,
                         gpus=[gpus] if gpus != None else None)
    # precision=16)
    trainer.fit(model, data_module)


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
