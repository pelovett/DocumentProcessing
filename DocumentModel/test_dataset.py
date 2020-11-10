from WOSDataset import WOSDataset
from WOSDataLoader import WOSDataModule

test_dirs = ['./data/WOS11967/', './data/WOS5736/', './data/WOS46985/', '~']


def create_dataset(directory):
    try:
        x = WOSDataset(file_path=directory)
        return True
    except FileNotFoundError:
        return False


def create_datamodule(directory):
    try:
        return WOSDataModule(directory, 'bert-base-uncased', 1)
    except:
        print('# Failed to create WOSDataModule')
        raise


def test_dataset_creation():
    assert create_dataset(test_dirs[0]) == True
    assert create_dataset(test_dirs[1]) == True
    assert create_dataset(test_dirs[2]) == True
    assert create_dataset(test_dirs[3]) == False


def test_dataset_iteration():
    x = WOSDataset(file_path=test_dirs[0])
    for i in x:
        sample = x[i]
        assert type(sample) == dict
        assert bool(sample)


def test_datamodule_creation():
    assert create_datamodule(test_dirs[0])
    assert create_datamodule(test_dirs[1])
    assert create_datamodule(test_dirs[2])


def test_prepare_data():
    x = WOSDataModule(test_dirs[0])
    x.prepare_data()


def test_train_dataloader():
    x = WOSDataModule(test_dirs[0])
    x.prepare_data()
    x.train_dataloader()


def test_val_dataloader():
    x = WOSDataModule(test_dirs[0])
    x.prepare_data()
    x.val_dataloader()


def test_test_dataloader():
    x = WOSDataModule(test_dirs[0])
    x.prepare_data()
    x.test_dataloader()
