import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

from keras import backend as K
from keras.models import load_model
from utility import LoadDataNoDefCW
from Model_NoDef import DFNet
import random
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
from conf_mat_plot import plot_confusion_matrice
import numpy as np
import argparse
import os

random.seed(0)

description = "Training and evaluating DF model for closed-world scenario on non-defended dataset"

print (description)
# Training the DF model
NB_EPOCH = 500   # Number of training epoch
print ("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 64 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 500 # Packet sequence length
OPTIMIZER = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # Optimizer

NB_CLASSES = 4 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)

parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--trial', default = 1, type = int, help='Trial number, 0-4 provided')
args = parser.parse_args()
trial = str(args.trial)
dataset="DC"



def confusion_matrix(pred, label):
    mat = np.zeros((NB_CLASSES, NB_CLASSES))

    # y-axis: label, x-axis:prediction
    for i in range(0, pred.shape[0]):
        mat[int(label[i]), int(pred[i])] = mat[int(label[i]), int(pred[i])] + 1

    # Uncomment if plotting the confusion matrix
    count = np.sum(mat, axis=1)
    count = np.where(count>np.zeros(count.shape), count, 1)
    for i in range(0, NB_CLASSES):
        mat[i, :] = mat[i, :]/count[i]

    return mat

# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test, _, _ = LoadDataNoDefCW(dataset, NB_CLASSES, trial)
# Please refer to the dataset format in readme

# # Convert data as float32 type
# X_train = X_train.astype('float32')
# X_valid = X_valid.astype('float32')
# X_test = X_test.astype('float32')
# y_train = y_train.astype('float32')
# y_valid = y_valid.astype('float32')
# y_test = y_test.astype('float32')

# # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
# X_train = X_train[:, :,np.newaxis]
# X_valid = X_valid[:, :,np.newaxis]
# X_test = X_test[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to categorical classes matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# Building and training model
print ("Building and training DF model")

model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])
print("Model compiled")

def lr_scheduler(epoch):
	if epoch % 20 == 0 and epoch != 0:
		lr = K.get_value(model.optimizer.lr)
		K.set_value(model.optimizer.lr, lr*0.13)
		print("lr changed to {}".format(lr*0.13))
	return K.get_value(model.optimizer.lr)


filepath = '/media/SATA_1/thilini_open_extra/final_codes/OpenMax/DC/closed/models/dc_closed_trial_'+trial+'.hdf5'


checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

if OPTIMIZER == 'SGD':
    scheduler = LearningRateScheduler(lr_scheduler)
    CallBacks.append(scheduler)

callbacks_list.append(EarlyStopping(monitor='val_loss', mode='min', patience=6))

model.summary()
# Start training

# Start training
history = model.fit(X_train, y_train,
		batch_size=BATCH_SIZE, epochs=NB_EPOCH,
		verbose=VERBOSE, validation_data=(X_valid, y_valid), callbacks=callbacks_list)


# Start evaluating model with testing data
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Testing accuracy:", score_test[1])


# clean up
del model

model = load_model(filepath)

# pred = model.predict(X_test)

# mat = confusion_matrix(np.argmax(pred, axis=1), np.argmax(y_test, axis=1))

# plot_confusion_matrice(mat, np.arange(0, NB_CLASSES), '/home/sec_user/thilini/Other_open/openmax/DC/closed/softmax_figs/confusion/'+str(trial)+'_closed_conf.png', NB_CLASSES)

# # Start evaluating model with testing data
# score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
# print("Testing accuracy:", score_test[1])


# # clean up
# del model