import gzip
import pickle
import os

def load_mnist_datasets(path):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' % path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')
        return train_set, val_set, test_set
