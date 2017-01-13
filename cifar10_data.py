import numpy as np
import random

def get_data():
    def to_onehot(targets):
        return np.eye(10)[targets.tolist()].squeeze()

    from keras.datasets import cifar10
    (images_train, labels_train), (images_test, labels_test) = cifar10.load_data()
    labels_train_onehot = to_onehot(labels_train)
    labels_test_onehot = to_onehot(labels_test)
    return (images_train, labels_train_onehot), (images_test, labels_test_onehot)

def get_batch(x, t, batch_size):
    n_samples = x.shape[0]
    k = random.randint(0, n_samples - batch_size - 1)
    return x[k: k + batch_size], t[k: k + batch_size]