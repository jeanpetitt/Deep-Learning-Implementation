import h5py
import numpy as np


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # train set features
    y_train = np.array(train_dataset["Y_train"][:]) # train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # test set features
    y_test = np.array(test_dataset["Y_test"][:]) # test set labels
    
    return X_train, y_train, X_test, y_test

# x_train, y_train, x_test, y_test = load_data()

# print(x_train)