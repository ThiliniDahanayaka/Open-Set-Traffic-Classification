import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

from openmax import Openmax

import numpy as np
import keras
import keras.backend as K
from keras.models import load_model
from closed.utility import LoadDataNoDefCW
import pickle
import math
import matplotlib
import sklearn.metrics
# from conf_mat_plot import plot_confusion_matrice
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
import json

n_closed = 40

parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--trial', default = 1, type = int, help='Trial number, 0-4 provided')
args = parser.parse_args()
trial = str(args.trial)
datasetName="IOT"


def accuracy(Inliers_true, other_true, Inliers_score_raw,  other_score_raw, minlist, num_classes):

    Inliers_pred = np.argmax(Inliers_score_raw, axis=1)
    # other_pred = np.argmax(other_score_raw, axis=1)

    Inliers_score = np.max(Inliers_score_raw, axis=1)
    # other_score = np.max(other_score_raw, axis=1)

    Inliers_label = []

    for i in range(0, Inliers_pred.shape[0]):
        if Inliers_score[i]>=minlist[int(Inliers_pred[i])]:
            Inliers_label.append(Inliers_pred[i])
        else:
            Inliers_label.append(num_classes)

    Inliers_label = np.array(Inliers_label)
    best_acc_inlier = np.sum(np.where(Inliers_label==Inliers_true, 1, 0))/Inliers_pred.shape[0]*100

    del Inliers_pred, Inliers_score, Inliers_score_raw

    other_pred = np.argmax(other_score_raw, axis=1)

    other_score = np.max(other_score_raw, axis=1)

    Outliers_label = []

    for i in range(0, other_pred.shape[0]):
        if other_score[i]>=minlist[int(other_pred[i])]:
            Outliers_label.append(other_pred[i])
        else:
            Outliers_label.append(num_classes)

    Outliers_label = np.array(Outliers_label)
    best_acc_outlier = np.sum(np.where(Outliers_label==other_true, 1, 0))/other_pred.shape[0]*100

    del other_pred, other_score, other_score_raw, minlist

    prediction = np.append(Inliers_label, Outliers_label)
    del Inliers_label, Outliers_label

    labels = np.append(Inliers_true, other_true)
    del Inliers_true, other_true

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


    return best_acc_inlier, best_acc_outlier, F

def get_small_set(x, y, n):
    ind = np.zeros((1,))
    for i in range(0, n_closed):
        inds = np.where(y==i)[0]
        ind = np.append(ind, inds[0:n], axis=0)

    ind = ind[1:].astype(int)
    return x[ind], y[ind]

def build(n_closed, filepath):
    from keras.models import Model
    from keras.layers import Dense, Layer, Input
    from keras.layers import Dropout, Activation, BatchNormalization, GlobalAveragePooling1D
    from keras.layers import Conv1D, MaxPooling1D 
    from keras.layers.advanced_activations import ELU
    from keras.optimizers import Adamax

    

    length = 475
    input_shape = (length, 1)
    
    inp = Input(shape=(length, 1))
    x1 = Conv1D(128, kernel_size=7, activation='tanh', input_shape=input_shape, use_bias=False, kernel_initializer='glorot_normal')(inp)
    x2 = BatchNormalization(axis=-1)(x1)
    x3 = MaxPooling1D(1)(x2)
    x4 = Dropout(rate=0.1)(x3)

    x5 = Conv1D(128, kernel_size=19, activation='elu', kernel_initializer='glorot_normal')(x4)
    x6 = BatchNormalization(axis=-1)(x5)
    x7 = MaxPooling1D(1)(x6)
    x8 = Dropout(rate=0.3)(x7)

    x9 = Conv1D(64, kernel_size=13, activation='elu', kernel_initializer='glorot_normal')(x8)
    x10 = BatchNormalization(axis=-1)(x9)
    x11 = MaxPooling1D(1)(x10)
    x12 = Dropout(rate=0.1)(x11)

    x13 = Conv1D(256, kernel_size=23, activation='selu', kernel_initializer='glorot_normal')(x12)
    x14 = BatchNormalization(axis=-1)(x13)
    x15 = MaxPooling1D(1)(x14)
    x16 = GlobalAveragePooling1D()(x15)

    x17 = Dense(180, activation='selu', kernel_initializer='glorot_normal')(x16)
    x18 = BatchNormalization()(x17)
    x19 = Dense(150, activation='selu', kernel_initializer='glorot_normal')(x18)
    x20 = BatchNormalization()(x19)
    x21 = Dense(n_closed, activation=None)(x20)
    x22 = Activation('softmax')(x21)

    model = Model(inputs=inp, outputs=[x21])

    opt = 'Adamax'

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.summary()
    model.load_weights(filepath)
    return model


# def confusion_matrix(pred, label):
#     mat = np.zeros((n_closed+1, n_closed+1))

#     # y-axis: label, x-axis:prediction
#     for i in range(0, pred.shape[0]):
#         mat[int(label[i]), int(pred[i])] = mat[int(label[i]), int(pred[i])] + 1

#     # Uncomment if plotting the confusion matrix
#     count = np.sum(mat, axis=1)
#     count = np.where(count>np.zeros(count.shape), count, 1)
#     for i in range(0, n_closed+1):
#         mat[i, :] = mat[i, :]/count[i]


#     return mat

# def micro_f_score(x):
#     # x is confusion matrix

#     TP1 = np.sum(np.diagonal(x))
#     TP = np.sum(np.diagonal(x[:-1, :]))
#     FP = np.sum(np.sum(x, axis=0))-TP1
#     FN = np.sum(np.sum(x, axis=1)) - TP1

#     print('TP:{} FP:{} FN:{}'.format(TP, FP, FN))

#     P = TP/(TP+FP)
#     R = TP/(TP+FN)
#     F = 2*P*R/(P+R)

#     return R




def getMinList(pred_y, y):
    min_list = []

    for c in range(0, n_closed):

        ind_c = np.where(y==c)[0]
        Inliers_score_raw = pred_y[ind_c]
        # print(c, Inliers_score_raw.shape)
        if Inliers_score_raw.shape[0]>0:
            Inliers_score = np.max(Inliers_score_raw, axis=1)
            Inliers_pred = np.argmax(Inliers_score_raw, axis=1)

            start = np.min(Inliers_score)
            end = np.max(Inliers_score)
            gap = 0.002 #(end- start)/200000 # precision:200000

            accuracy_thresh = 90.0 
            accuracy_range = np.arange(start, end, gap)
            for i, delta in enumerate(accuracy_range):
                Inliers_label = np.where(Inliers_score>=delta, Inliers_pred, n_closed)
                y_ = y[ind_c]
                # print(Inliers_label.shape, y_.shape)
                a = np.sum(np.where(y_ == Inliers_label, 1, 0))/Inliers_label.shape[0]*100  

                if i==0 and a<accuracy_thresh:
                    print('Closed set accuracy did not reach ', accuracy_thresh, a)
                    min_list.append(delta)

                elif a<accuracy_thresh and i>0:
                    delta = accuracy_range[i-1]
                    min_list.append(delta)

                elif i==len(accuracy_range)-1:
                    print('Closed set accuracy did not fall below ', accuracy_thresh, a)
                    min_list.append(delta)

        else:
            (min_list.append(0.1))

    return (np.array(min_list))

dataXr, y_train, _, _, _, _, _, _ = LoadDataNoDefCW(datasetName, n_closed, trial_num=trial)
# dataXr, y_train = get_small_set(dataXr, y_train, 200)
# dataXr = dataXr[:, 0:1500, np.newaxis]


model = load_model("/media/SATA_1/thilini_open_extra/final_codes/OpenMax/IOT/closed/models/iot_closed_trial_"+trial+".hdf5")
pred = model.predict(dataXr)


model_path =  "/media/SATA_1/thilini_open_extra/final_codes/OpenMax/IOT/closed/models/iot_closed_trial_"+trial+".hdf5"
sub_model = build(n_closed, model_path)
prenultimate_layer_out = sub_model.predict(dataXr)


openmax_ob = Openmax(alpharank=1, tailsize=5, decision_dist_fn='euclidean')
openmax_ob.update_class_stats(prenultimate_layer_out, pred, keras.utils.to_categorical(y_train))
print('updated class data')


_, _, dataXr, y_train, _, _, _, _ = LoadDataNoDefCW(datasetName, n_closed, trial_num=trial)

dataXr, y_train = get_small_set(dataXr, y_train, 100)
# dataXr = dataXr[:, 0:1500, np.newaxis]

prenultimate_layer_out = sub_model.predict(dataXr)
open_prob = openmax_ob.predict_prob_open(prenultimate_layer_out)


min_list = getMinList(open_prob, y_train)

print('minlist created')

_, _, _, _, xtest, ytest, _, _ = LoadDataNoDefCW(datasetName, n_closed, trial_num=trial)

logits_test = sub_model.predict(xtest)
open_prob_c = openmax_ob.predict_prob_open(logits_test)

del logits_test, xtest

_, _, _, _, _, _, openx, openy = LoadDataNoDefCW(datasetName, n_closed, trial_num=trial)

logits_test = sub_model.predict( openx)
open_prob_o = openmax_ob.predict_prob_open(logits_test)

del logits_test, openx

closed_acc, open_acc, f1 = accuracy(ytest, openy, open_prob_c,  open_prob_o, min_list, n_closed)
print('closed acc:{}'.format(closed_acc))
print('open acc:{}'.format(open_acc))
print('f_score:{}'.format(f1))










