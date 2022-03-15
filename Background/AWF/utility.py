import pickle
import numpy as np
from numpy import load

# Load data for non-defended dataset for CW setting
def LoadDataNoDefCW():

    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = "/media/SATA_1/thilini_open_extra/final_codes/Background/AWF/dataset/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    
    X_train = load(dataset_dir+'X_train.npy')
    X_train = X_train[:, 0:1500]
    y_train = load(dataset_dir+'y_train.npy')

    # Load validation data
    X_valid = load(dataset_dir+'X_valid.npy')
    X_valid = X_valid[:, 0:1500]
    y_valid = load(dataset_dir+'y_valid.npy')

    # Load testing data
    X_test = load(dataset_dir+'X_test.npy')
    X_test = X_test[:, 0:1500]
    y_test = load(dataset_dir+'y_test.npy')

    # Load testing data
    X_open = load(dataset_dir+'X_test_open.npy')
    X_open = X_open[:, 0:1500]
    y_open = load(dataset_dir+'y_test_open.npy')

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)
    print ("X: OPen data's shape : ", X_open.shape)
    print ("y: OPen data's shape : ", y_open.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_open, y_open
