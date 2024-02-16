
import numpy as np
from keras.utils import np_utils
import json
import pickle
from numpy import load
from sklearn.utils import shuffle

def normalize(X,epsilon=1e-8):
		
    Raws=X.shape[0]
    Columns=X.shape[1]
    for i in range(Raws):
        Tot=0
        for j in X[i]:
            Tot+=j
            mean=Tot/Columns

        Dis=0
        for j in X[i]:
            Dis+=(j-mean)**2
            Var=Dis/Columns

        X[i]=(X[i]-mean)/(np.sqrt(Var)+epsilon)
                                        
    return X

# # Load data for non-defended dataset for CW setting
def LoadDataIot():

    dataset_dir = '/home/Yasod/Yasod/SETA/dataset/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    X_train = load(dataset_dir+'X_train_5.npy',allow_pickle=True)
    y_train = load(dataset_dir+'y_train_5.npy',allow_pickle=True)

    
    # Load validation data
    X_valid = load(dataset_dir+'X_valid_5.npy',allow_pickle=True)
    y_valid = load(dataset_dir+'y_valid_5.npy',allow_pickle=True)

    # Load testing data
    X_test = load(dataset_dir+'X_test_5.npy',allow_pickle=True)
    y_test = load(dataset_dir+'y_test_5.npy',allow_pickle=True)

    X_train, y_train = shuffle(X_train, y_train)
        
    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

