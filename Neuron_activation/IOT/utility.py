import pickle
import numpy as np
import json

# Load data for non-defended dataset for CW setting

def LoadDataNoDefCW(trial_num="1", n_closed=40):
    datasetName = "IOT"
    if datasetName == "IOT":
        datatpath = '/media/SATA_1/thilini_open_extra/final_datasets/IOT/'
    else:
        print('unknown dataset')
        return

    with open("/media/SATA_1/thilini_open_extra/final_datasets/IOT/data_file.json") as config_file:
        cfg = json.load(config_file)[trial_num]
    closeset = cfg["closed"]
    openset = cfg["open"]

    # print('len open = {}'.format(len(openset)))
    # print('len closed = {}'.format(len(closeset)))
        
    x = np.load(datatpath+'X_train.npy')
    y = np.load(datatpath+'y_train.npy')

    y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(closeset):
        ind = np.where(y==val)[0]
        x_new = np.append(x_new, x[ind], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((len(ind),))*count)

    x_new = x_new[1:]
    y_new = y_new[1:]

    del x, y

    x_new = x_new.astype('float32')
    y_train = y_new.astype('float32')


    x_train = x_new[:, 0:475, np.newaxis]



    x = np.load(datatpath+'X_valid.npy')
    y = np.load(datatpath+'y_valid.npy')

    y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(closeset):
        ind = np.where(y==val)[0]
        x_new = np.append(x_new, x[ind], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((len(ind),))*count)

    x_new = x_new[1:]
    y_new = y_new[1:]

    del x, y

    x_new = x_new.astype('float32')
    y_valid = y_new.astype('float32')


    x_valid = x_new[:, 0:475, np.newaxis]

    x = np.load(datatpath+'X_test.npy')
    y = np.load(datatpath+'y_test.npy')

    y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(closeset):
        ind = np.where(y==val)[0]
        x_new = np.append(x_new, x[ind], axis=0)
        # print(x_new.shape)
        y_new = np.append(y_new, np.ones((len(ind),))*count)

    x_new = x_new[1:]
    y_new = y_new[1:]

    del x, y

    x_new = x_new.astype('float32')
    y_test = y_new.astype('float32')


    x_test = x_new[:, 0:475, np.newaxis]


    open_l = n_closed
    x = np.load(datatpath+'X_test.npy')
    y = np.load(datatpath+'y_test.npy')

    # y_new = np.ones((1,))
    x_new = np.ones((1, x.shape[1]))

    for count, val in enumerate(openset):
        ind = np.where(y==val)[0]
        x_new = np.append(x_new, x[ind][0:138], axis=0)
        # print(x_new.shape)

    x_new = x_new[1:]
    y_new = np.ones((x_new.shape[0]))*open_l

    del x, y

    x_new = x_new.astype('float32')
    y_open = y_new.astype('float32')


    x_open = x_new[:, 0:475, np.newaxis]

    # unknownSet=None

    return x_train, y_train, x_valid, y_valid, x_test, y_test, x_open, y_open

# def LoadDataNoDefCW():

#     print ("Loading non-defended dataset for closed-world scenario")
#     # Point to the directory storing data
#     dataset_dir ="/home/sec_user/thilini/Anchor_loss/IOT/dataset/"
#     # y represents a sequence of corresponding label (website's label)

#     # Load training data
#     X_train = np.load(dataset_dir+'X_train.npy')
#     X_train = X_train[:, 0:1500]
#     y_train = np.load(dataset_dir+'y_train.npy')

#     # Load validation data
#     X_valid = np.load(dataset_dir+'X_valid.npy')
#     X_valid = X_valid[:, 0:1500]
#     y_valid = np.load(dataset_dir+'y_valid.npy')

#     # Load testing data
#     X_test = np.load(dataset_dir+'X_test.npy')
#     X_test = X_test[:, 0:1500]
#     y_test = np.load(dataset_dir+'y_test.npy')

#     print ("Data dimensions:")
#     print ("X: Training data's shape : ", X_train.shape)
#     print ("y: Training data's shape : ", y_train.shape)
#     print ("X: Validation data's shape : ", X_valid.shape)
#     print ("y: Validation data's shape : ", y_valid.shape)
#     print ("X: Testing data's shape : ", X_test.shape)
#     print ("y: Testing data's shape : ", y_test.shape)

#     return X_train, y_train, X_valid, y_valid, X_test, y_test
