
import numpy as np
from tensorflow.keras.utils import to_categorical
import json
import _pickle as cPickle


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
    Length=1500
    
    dataset_dir = "/home/Yasod/Yasod/DF/dataset/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    
    with open( dataset_dir+"X_train_NoDef.pkl","rb") as f:
        X_train = cPickle.load(f,encoding='latin1')
    X_train = np.array(X_train)
    X_train = X_train[:, 0:Length]
    with open( dataset_dir+"y_train_NoDef.pkl","rb") as f:
        y_train = cPickle.load(f,encoding='latin1')
    y_train = np.array(y_train)
    
    
    # Load validation data
    with open( dataset_dir+"X_valid_NoDef.pkl","rb") as f:
        X_valid = cPickle.load(f,encoding='latin1')
    X_valid = np.array(X_valid)
    X_valid = X_valid[:, 0:Length]
    with open( dataset_dir+"y_valid_NoDef.pkl","rb") as f:
        y_valid = cPickle.load(f,encoding='latin1')
    y_valid = np.array(y_valid)
    
    # Load testing data
    with open( dataset_dir+"X_test_NoDef.pkl","rb") as f:
        X_test = cPickle.load(f,encoding='latin1')
    X_test = np.array(X_test)
    X_test = X_test[:, 0:Length]
    with open( dataset_dir+"y_test_NoDef.pkl","rb") as f:
        y_test = cPickle.load(f,encoding='latin1')
    y_test = np.array(y_test)


    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def LoadDataIot_background():

    dataset_dir = "/home/Yasod/Yasod/DF/dataset/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    
    with open( dataset_dir+"X_train_NoDef.pkl","rb") as f:
        X_train = cPickle.load(f,encoding='latin1')
    X_train = np.array(X_train)
    X_train = X_train[:, 0:5000]
    with open( dataset_dir+"y_train_NoDef.pkl","rb") as f:
        y_train = cPickle.load(f,encoding='latin1')
    y_train = np.array(y_train)
    
    
    # Load validation data
    with open( dataset_dir+"X_valid_NoDef.pkl","rb") as f:
        X_valid = cPickle.load(f,encoding='latin1')
    X_valid = np.array(X_valid)
    X_valid = X_valid[:, 0:5000]
    with open( dataset_dir+"y_valid_NoDef.pkl","rb") as f:
        y_valid = cPickle.load(f,encoding='latin1')
    y_valid = np.array(y_valid)
    
    # Load testing data
    with open( dataset_dir+"X_test_NoDef.pkl","rb") as f:
        X_test = cPickle.load(f,encoding='latin1')
    X_test = np.array(X_test)
    X_test = X_test[:, 0:5000]
    with open( dataset_dir+"y_test_NoDef.pkl","rb") as f:
        y_test = cPickle.load(f,encoding='latin1')
    y_test = np.array(y_test)
    
    with open( dataset_dir+"X_open.pkl","rb") as f:
        X_open = cPickle.load(f,encoding='latin1')
    X_open = np.array(X_open)
    X_open = X_open[:, 0:5000]
    
    X_train = np.concatenate((X_train,X_open[:len(X_train)]),axis=0)
    X_open=X_open[len(X_train):]
    y_train = np.concatenate((y_train,[NB_CLASSES]*len(y_train)),axis=0)
    X_train,y_train=shuffle(X_train, y_train)
    
    X_valid = np.concatenate((X_valid,X_open[:len(X_valid)]),axis=0)
    X_open=X_open[len(X_valid):]
    y_valid = np.concatenate((y_valid,[NB_CLASSES]*len(y_valid)),axis=0)
    X_valid,y_valid=shuffle(X_valid, y_valid)
    
    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)
    
    np.save(dataset_dir+'X_open_rest.npy',X_open)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
