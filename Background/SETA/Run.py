import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

from keras import backend as K
from utility import LoadDataIot
from Model_cnn import DCNet
import random
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np
# import pickle
import argparse
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(0)




parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--trial', default = 1, type = int, help='Trial number, 0-4 provided')
args = parser.parse_args()
trial = str(args.trial)


# # if modelName=='cnn':
# #     NB_EPOCH = 500   
# #     BATCH_SIZE = 70 # Batch size
# #     VERBOSE = 2 # Output display mode
# #     OPTIMIZER = 'Adamax' # Optimizer
# #     data_dim = 475
# #     INPUT_SHAPE = (data_dim,1)
# #     from Model_cnn import DCNet
# #     NB_CLASSES = 41

# # elif modelName=='cudnnLstm':
# #     NB_EPOCH = 500   
# #     BATCH_SIZE = 130 # Batch size
# #     VERBOSE = 2 # Output display mode
# #     OPTIMIZER = 'Adamax' # Optimizer
# #     data_dim = 350
# #     INPUT_SHAPE = (data_dim,1)
# #     from Model_LSTM import DCNet

# # def reshapeData(allData, modelName):
# #     if 'cnn' == modelName:
# #         allData = np.expand_dims(allData, axis=2)
# #     elif 'lstm' == modelName:
# #         allData = allData.reshape(allData.shape[0], allData.shape[1], 1)
# #     elif 'cudnnLstm' == modelName:
# #         allData = allData[:, 0:data_dim]
# #         allData = allData.reshape(allData.shape[0], allData.shape[1], 1)
# #     elif 'sae' == modelName:
# #         return allData
# #     elif 'ensemble' == modelName:
# #         return allData
# #     else:
# #         raise ValueError('model name {} is not defined'.format(modelName))

# #     return allData


# # def processData(modelName, X_train, y_train, X_test, y_test, NUM_CLASS):
# #     X_train = reshapeData(X_train, modelName)
# #     X_test = reshapeData(X_test, modelName)
# #     y_train = np_utils.to_categorical(y_train, NUM_CLASS)
# #     y_test = np_utils.to_categorical(y_test, NUM_CLASS)
# #     return X_train, y_train, X_test, y_test


# # Data: shuffled and split between train and test sets
# print ("Loading and preparing data for training, and evaluating the model")
# X_train, y_train, X_valid, y_valid, X_test, y_test, X_open, y_open  = LoadDataIot(trial_num=trial, num_classes=NB_CLASSES-1)

# y_train = np_utils.to_categorical(y_train, NB_CLASSES)
# y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
# y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# y_open = np_utils.to_categorical(y_open, NB_CLASSES)

# filepath = '/home/sec_user/thilini/Other_open/background/IOT/models/iot_'+trial+'.hdf5'


# # X_train, y_train, X_test, y_test = processData(modelName, X_train, y_train, X_test, y_test, NB_CLASSES)

# print(X_train.shape, 'train shape')
# print(X_test.shape, 'test shape')

# # Building and training model
# print("Building and training DF model")

# model = DCNet.build(input_shape=INPUT_SHAPE, nb_classes=NB_CLASSES)

# model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
# 	metrics=["accuracy"])
# print("Model compiled")

# def lr_scheduler(epoch):
# 	if epoch % 20 == 0 and epoch != 0:
# 		lr = K.get_value(model.optimizer.lr)
# 		K.set_value(model.optimizer.lr, lr*0.13)
# 		print("lr changed to {}".format(lr*0.13))
# 	return K.get_value(model.optimizer.lr)


# # filepath = '/home/sec-user/thilini/New_CNN_for_TMC/defense/models/NO-defence.hdf5'


# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# if OPTIMIZER == 'SGD':
#     scheduler = LearningRateScheduler(lr_scheduler)
#     CallBacks.append(scheduler)

# callbacks_list.append(EarlyStopping(monitor='val_acc', mode='max', patience=6))

# model.summary()
# # Start training
# history = model.fit(X_train, y_train,
# 		batch_size=BATCH_SIZE, epochs=NB_EPOCH,
# 		verbose=VERBOSE, validation_data=[X_valid, y_valid], callbacks=callbacks_list)
    

# # Start evaluating model with testing data
# score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
# print("Closed Testing accuracy:", score_test[1])

# # Start evaluating model with testing data
# score_test = model.evaluate(X_open, y_open, verbose=VERBOSE)
# print("Open Testing accuracy:", score_test[1])

# pred_c = model.predict(X_test)
# y_test = np.argmax(y_test, axis=1)

# pred_o = model.predict(X_open)
# y_open = np.argmax(y_open, axis=1)
# print(y_open.max(), y_open.min())

# auroc = auroc(pred_c, pred_o, title='IOT_background', trial_num=trial)
# print('AUROC:', auroc)

# # clean up
# del model


def get_f1(prediction, labels, num_classes):
    

    # calculate confusion matrix
    mat = np.zeros((num_classes+1, num_classes+1))

    # y-axis: label, x-axis:prediction
    for i in range(prediction.shape[0]):
        mat[int(labels[i]), int(prediction[i])] = mat[int(labels[i]), int(prediction[i])] + 1

    P=0
    R=0
    for c in range(0, num_classes):
        tp = np.diagonal(mat)[c]
        fp = np.sum(mat[:, c])-tp
        fn = np.sum(mat[c, :])-tp
        # print('class:{}, tp:{}, fp:{}, fn:{}'.format(c, tp, fp, fn))

        if (tp+fp==0):
            P = P
        else:
            P = P+(tp/(tp+fp))

        if (tp+fn==0):
            R = R
        else:
            R = R+(tp/(tp+fn))



    P = P/num_classes
    R = R/num_classes

    F = 2*P*R/(P+R)

    return F




# Training the DF model
NB_EPOCH = 500   # Number of training epoch
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 64 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 500 # Packet sequence length
OPTIMIZER = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # Optimizer

NB_CLASSES = 9 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test, X_open, y_open  = LoadDataIot(trial_num=trial, num_classes=NB_CLASSES-1)
# Please refer to the dataset format in readme

# x_mean = np.mean(X_train, axis=0)
# x_std = np.std(X_train, axis=0)

# X_train = (X_train-x_mean)/x_std
# X_valid = (X_valid-x_mean)/x_std
# X_test = (X_test-x_mean)/x_std
# X_open = (X_open-x_mean)/x_std

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
y_open = np_utils.to_categorical(y_open, NB_CLASSES)

# Building and training model
print("Building and training DF model")

model = DCNet.build(input_shape=INPUT_SHAPE, nb_classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
    metrics=["accuracy"])
print("Model compiled")

filepath = '/media/SATA_1/thilini_open_extra/final_codes/Background/DC/models/seta_'+trial+'.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

callbacks_list.append(EarlyStopping(monitor='val_acc', mode='max', patience=20))

model.summary()

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(X_valid.shape)
# print(y_valid.shape)
# print(X_open.shape)
# print(y_open.shape)

history = model.fit(X_train, y_train,
        batch_size=BATCH_SIZE, epochs=NB_EPOCH,
        verbose=VERBOSE, validation_data=[X_valid, y_valid], callbacks=callbacks_list)
    

# Start evaluating model with testing data
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Closed Testing accuracy:", score_test[1])

# Start evaluating model with testing data
score_test = model.evaluate(X_open, y_open, verbose=VERBOSE)
print("Open Testing accuracy:", score_test[1])

pred_c = model.predict(X_test)
pred_o = model.predict(X_open)
pred = np.append(np.argmax(pred_c, axis=1), np.argmax(pred_o, axis=1))
labels = np.append(np.argmax(y_test, axis=1), np.argmax(y_open, axis=1))

print('F1:', get_f1(pred, labels, NB_CLASSES-1))
