
import numpy as np
from keras.utils import np_utils
import json
import pickle

# # Load data for non-defended dataset for CW setting
# def LoadDataIot():

#     print ("Loading non-defended dataset for closed-world scenario")
#     # Point to the directory storing data
#     dataset_dir = "/media/SATA_4/Smart_speaker_WiSec2020/DeepVCFingerprinting-master/attack/experiments/temp_test_train_split/"


#     # Load training data
#     X_train = np.load(dataset_dir+'X_train.npy')
#     y_train = np.load(dataset_dir+'y_train.npy')

#     # Load testing data
#     X_test = np.load(dataset_dir+'X_test.npy')
#     y_test = np.load(dataset_dir+'y_test.npy')

#     print ("Data dimensions:")
#     print ("X: Training data's shape : ", X_train.shape)
#     print ("y: Training data's shape : ", y_train.shape)
#     print ("X: Testing data's shape : ", X_test.shape)
#     print ("y: Testing data's shape : ", y_test.shape)

#     return X_train, y_train, X_test, y_test

def LoadDataIot(trial_num="1", num_classes=4):
    datatpath = '/media/SATA_1/thilini_open_extra/final_datasets/DC/'

    with open("/media/SATA_1/thilini_open_extra/final_codes/Background/DC/data_file.json") as config_file:
        cfg = json.load(config_file)[trial_num]
    closeset = cfg["closed"]
    openset = cfg["open"]

    # x = np.load(datatpath+'X_train.npy')
    # y = np.load(datatpath+'y_train.npy')

    with open(datatpath+'video_X_train.pkl', 'rb') as file:
        x = pickle.load(file)
    with open(datatpath+'video_y_train.pkl', 'rb') as file:
        y = pickle.load(file)

    y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(closeset):
        ind = np.where(y==val)[0]
        # print(val, len(ind))
        x_new = np.append(x_new, x[ind], axis=0)
        y_new = np.append(y_new, np.ones((len(ind),))*count)

    for count1, val in enumerate(openset[0:2]):
        ind = np.where(y==val)[0]
        # print(val, len(ind))
        x_new = np.append(x_new, x[ind[0:200]], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((200,))*num_classes)

    x_new = x_new[1:]
    y_new = y_new[1:]

    del x, y

    x_new = x_new.astype('float32')
    y_new = y_new.astype('float32')


    x_new = x_new[:, 0:500, np.newaxis]
    # x_new = np.swapaxes(x_new, 2, 1)
    
    x_train = x_new
    y_train = y_new

    # x = np.load(datatpath+'X_valid.npy')
    # y = np.load(datatpath+'y_valid.npy')

    with open(datatpath+'video_X_valid.pkl', 'rb') as file:
        x = pickle.load(file)
    with open(datatpath+'video_y_valid.pkl', 'rb') as file:
        y = pickle.load(file)

    y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(closeset):
        ind = np.where(y==val)[0]
        # print(val, len(ind))
        x_new = np.append(x_new, x[ind], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((len(ind),))*count)

    for count1, val in enumerate(openset[0:2]):
        ind = np.where(y==val)[0]
        # print(val, len(ind))
        x_new = np.append(x_new, x[ind[0:2]], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((2,))*num_classes)

    x_new = x_new[1:]
    y_new = y_new[1:]

    del x, y

    x_new = x_new.astype('float32')
    y_new = y_new.astype('float32')


    x_new = x_new[:, 0:500, np.newaxis]
    # x_new = np.swapaxes(x_new, 2, 1)
    
    x_valid = x_new
    y_valid = y_new

    # x = np.load(datatpath+'X_test.npy')
    # y = np.load(datatpath+'y_test.npy')

    with open(datatpath+'video_X_test.pkl', 'rb') as file:
        x = pickle.load(file)
    with open(datatpath+'video_y_test.pkl', 'rb') as file:
        y = pickle.load(file)

    y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(closeset):
        ind = np.where(y==val)[0]
        # print(val, len(ind))
        x_new = np.append(x_new, x[ind], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((len(ind),))*count)

    # for count1, val in enumerate(openset[0:2]):
    #     ind = np.where(y==val)[0]
    #     # print(val, len(ind))
    #     x_new = np.append(x_new, x[ind[0:30]], axis=0)
    #     # print(x_new.shape)
    #     y_new = np.append(y_new, np.ones((30,))*num_classes)
    #     # print(y_new.shape)

    x_new = x_new[1:]
    y_new = y_new[1:]

    del x, y

    x_new = x_new.astype('float32')
    y_new = y_new.astype('float32')


    x_new = x_new[:, 0:500, np.newaxis]
    # x_new = np.swapaxes(x_new, 2, 1)

    x_test = x_new
    y_test = y_new


    # open_l = cfg_o['num_known_classes']
    open_l = num_classes
    # x = np.load(datatpath+'X_test.npy')
    # y = np.load(datatpath+'y_test.npy')

    with open(datatpath+'video_X_test.pkl', 'rb') as file:
        x = pickle.load(file)
    with open(datatpath+'video_y_test.pkl', 'rb') as file:
        y = pickle.load(file)

    # y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(openset[2:]):
        ind = np.where(y==val)[0]
        # print(val, len(ind))
        x_new = np.append(x_new, x[ind][0:60], axis=0)
        # print(x_new.shape)

    x_new = x_new[1:]
    y_new = np.ones((x_new.shape[0]))*open_l

    del x, y

    x_new = x_new.astype('float32')
    y_new = y_new.astype('float32')


    x_new = x_new[:, 0:500, np.newaxis]
    # x_new = np.swapaxes(x_new, 2, 1)
    
    x_open = x_new
    y_open = y_new

    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_valid.shape)
    # print(x_open.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # print(y_valid.shape)
    # print(y_open.shape)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, x_open, y_open


def Load_valid(trial_num="1", num_classes=4):
    datatpath = '/home/sec_user/thilini/Anchor_loss/DC/dataset/'

    with open("/home/sec_user/thilini/Other_open/background/DC/data_file.json") as config_file:
        cfg = json.load(config_file)[trial_num]
    closeset = cfg["closed"]
    openset = cfg["open"]

    with open(datatpath+'video_X_test.pkl', 'rb') as file:
        x = pickle.load(file)
    with open(datatpath+'video_y_test.pkl', 'rb') as file:
        y = pickle.load(file)

    y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count1, val in enumerate(openset[0:2]):
        ind = np.where(y==val)[0]
        # print(val, len(ind))
        x_new = np.append(x_new, x[ind[0:30]], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((30,))*num_classes)

    x_new = x_new[1:]
    y_new = y_new[1:]

    del x, y

    x_new = x_new.astype('float32')
    y_new = y_new.astype('float32')


    x_new = x_new[:, 0:500, np.newaxis]
    # x_new = np.swapaxes(x_new, 2, 1)
    
    x_valid = x_new
    y_valid = y_new

    return x_valid, y_valid


def get_reduced_data_to_tranfer(X_train, y_train, X_test, y_test, n_transfered):
	ind_tr = np.where(y_train<n_transfered)[0]
	ind_te = np.where(y_test<n_transfered)[0]

	return X_train[ind_tr], y_train[ind_tr], X_test[ind_te], y_test[ind_te]

def get_reduced_data_to_learn(X_train, y_train, X_test, y_test, n_transfered):
	ind_tr = np.where(y_train>=n_transfered)[0]
	ind_te = np.where(y_test>=n_transfered)[0]

	return X_train[ind_tr], y_train[ind_tr]-n_transfered, X_test[ind_te], y_test[ind_te]-n_transfered


def get_classwise_data(x, y, traces, NB_CLASSES):
    y = y.argmax(axis=1)
    tr = np.zeros((1,))

    for c in range(0, NB_CLASSES):
        ind = np.where(y==c)[0]
        # print(len(ind))
        tr = np.append(tr, ind[0:traces])

    tr = tr[1:].astype(int)

    ind = np.arange(len(tr))
    np.random.shuffle(ind)
    tr=tr[ind]

    return x[tr], np_utils.to_categorical(y[tr],NB_CLASSES) 
    
    
def reshapeData(allData, modelName):
    if 'cnn' == modelName:
        allData = np.expand_dims(allData, axis=2)
    elif 'lstm' == modelName:
        allData = allData.reshape(allData.shape[0], allData.shape[1], 1)
    elif 'cudnnLstm' == modelName:
        allData = allData.reshape(allData.shape[0], allData.shape[1], 1)
    elif 'sae' == modelName:
        return allData
    elif 'ensemble' == modelName:
        return allData
    else:
        raise ValueError('model name {} is not defined'.format(modelName))

    return allData


def processData(modelName, X_train, y_train, X_test, y_test, NUM_CLASS):
    X_train = reshapeData(X_train, modelName)
    X_test = reshapeData(X_test, modelName)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)
    return X_train, y_train, X_test, y_test

# LoadDataIot(trial_num=str(1))