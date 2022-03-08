import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

from openmax import Openmax

import numpy as np
import keras
import keras.backend as K
from keras.models import load_model
import pickle
import math
import matplotlib
import sklearn.metrics
from closed.utility import LoadDataNoDefCW
# from conf_mat_plot import plot_confusion_matrice
matplotlib.use('Agg')

import matplotlib.pyplot as plt

n_closed = 200

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

    
    

# def auroc(inData, outData, title='test', trial_num=1):
#     # print(inData.shape, outData.shape)
#     inDataMin = np.max(inData, 1)
#     outDataMin = np.max(outData, 1)

#     Y1 = outDataMin
#     X1 = inDataMin
#     end = np.max([np.max(X1), np.max(Y1)])
#     start = np.min([np.min(X1),np.min(Y1)])
#     # end = np.min(X1)
#     # start = np.max(X1)
#     gap = (end- start)/200000

#     print(start, end, gap)

#     aurocBase = 0.0
#     fprTemp = 1.0
#     for delta in np.arange(start, end, gap):
#         tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
#         fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
#         # f1.write("{}\n".format(tpr))
#         # f2.write("{}\n".format(fpr))
#         aurocBase += (-fpr+fprTemp)*tpr
#         # print('delta:{}, tpr:{}, fpr:{}, fprTemp:{}, aurocBase:{}'.format(delta,    tpr, fpr, fprTemp, aurocBase))
#         fprTemp = fpr

#     return aurocBase

# def get_f1(Inliers_true, other_true, Inliers_score_raw,  other_score_raw, delta, num_classes):
#     '''
#     delta: threshold value to reject open samples calculated from method 'accuracy'
#     '''

#     # print(Inliers_true.shape, other_true.shape)
#     Inliers_pred = np.argmax(Inliers_score_raw, axis=1)
#     other_pred = np.argmax(other_score_raw, axis=1)

#     Inliers_score = np.max(Inliers_score_raw, axis=1)
#     other_score = np.max(other_score_raw, axis=1)

#     Inliers_label = np.where(Inliers_score>=delta, Inliers_pred, num_classes)
#     Outliers_label = np.where(other_score>=delta, other_pred, num_classes)

#     prediction = np.append(Inliers_label, Outliers_label, axis=0)
#     labels = np.append(Inliers_true, other_true, axis=0)

#     # calculate confusion matrix
#     mat = np.zeros((num_classes+1, num_classes+1))

#     # y-axis: label, x-axis:prediction
#     for i in range(prediction.shape[0]):
#         mat[int(labels[i]), int(prediction[i])] = mat[int(labels[i]), int(prediction[i])] + 1

#     P=0
#     R=0
#     for c in range(0, num_classes):
#         tp = np.diagonal(mat)[c]
#         fp = np.sum(mat[:, c])-tp
#         fn = np.sum(mat[c, :])-tp
#         # print('class:{}, tp:{}, fp:{}, fn:{}'.format(c, tp, fp, fn))

#         P = P+(tp/(tp+fp))
#         R = R+(tp/(tp+fn))



#     P = P/num_classes
#     R = R/num_classes

#     F = 2*P*R/(P+R)

    return F

def get_small_set(x, y, n):
    ind = np.zeros((1,))
    for i in range(0, n_closed):
        inds = np.where(y==i)[0]
        ind = np.append(ind, inds[0:n], axis=0)

    ind = ind[1:].astype(int)
    return x[ind], y[ind]

def build(n_closed, filepath):
    from keras.models import Model
    from keras.layers import Dense, Layer, Input, Flatten
    from keras.layers import Dropout, Activation, BatchNormalization
    from keras.layers import Conv1D, MaxPooling1D 
    from keras.layers.advanced_activations import ELU
    from keras.optimizers import Adamax

    filter_num = ['None',32,64,128,256]
    kernel_size = ['None',8,8,8,8]
    conv_stride_size = ['None',1,1,1,1]
    pool_stride_size = ['None',4,4,4,4]
    pool_size = ['None',8,8,8,8]

    length = 1500
    input_shape = (length, 1)
    
    inp = Input(shape=(length, 1))
    x1 = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv1')(inp)
    x2 = BatchNormalization(axis=-1)(x1)
    x3 = ELU(alpha=1.0, name='block1_adv_act1')(x2)
    x4 = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv2')(x3)
    x5 = BatchNormalization(axis=-1)(x4)
    x6 = ELU(alpha=1.0, name='block1_adv_act2')(x5)
    x7 = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block1_pool')(x6)
    x8 = Dropout(0.1, name='block1_dropout')(x7)

    x9 = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1')(x8)
    x10 = BatchNormalization()(x9)
    x11 = Activation('relu', name='block2_act1')(x10)

    x12 = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv2')(x11)
    x13 = BatchNormalization()(x12)
    x14 = Activation('relu', name='block2_act2')(x13)
    x15 = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool')(x14)
    x16 = Dropout(0.1, name='block2_dropout')(x15)

    x17 = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv1')(x16)
    x18 = BatchNormalization()(x17)
    x19 = Activation('relu', name='block3_act1')(x18)
    x20 = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv2')(x19)
    x21 = BatchNormalization()(x20)
    x22 = Activation('relu', name='block3_act2')(x21)
    x23 = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool')(x22)
    x24 = Dropout(0.1, name='block3_dropout')(x23)

    x25 = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv1')(x24)
    x26 = BatchNormalization()(x25)
    x27 = Activation('relu', name='block4_act1')(x26)
    x28 = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv2')(x27)
    x29 = BatchNormalization()(x28)
    x30 = Activation('relu', name='block4_act2')(x29)
    x31 = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                           padding='same', name='block4_pool')(x30)
    x32 = Dropout(0.1, name='block4_dropout')(x31)

    x33 = Flatten(name='flatten')(x32)
    x34 = Dense(512,  name='fc1')(x33)
    x35 = BatchNormalization()(x34)
    x36 = Activation('relu', name='fc1_act')(x35)

    x37 = Dropout(0.7, name='fc1_dropout')(x36)

    x38 = Dense(512, name='fc2')(x37)
    x39 = BatchNormalization()(x38)
    x40 = Activation('relu', name='fc2_act')(x39)

    x41 = Dropout(0.5, name='fc2_dropout')(x40)

    x42 = Dense(n_closed, name='fc3')(x41)
    x43 = Activation('softmax', name="softmax")(x42)

    model = Model(inputs=inp, outputs=[x42])

    opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

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

# def accuracy(pred, label, thresh, thresh_type=0):
#     if thresh_type==1:
#         label = np.where(label<n_closed, label, n_closed)
#         pred_l = np.argmax(pred, axis=1)
#         for i in range(0, label.shape[0]):
#             if pred_l[i]<n_closed:
#                 if pred[i, pred_l[i]]<thresh[pred_l[i]]:
#                     pred_l[i] = n_closed
            
    
#         a = np.sum(np.where(label == pred_l, 1, 0))/pred.shape[0]*100
#         return a, label, pred_l

#     else:
#         label = np.where(label<n_closed, label, n_closed)
#         pred = np.where(np.max(pred, axis=1)>thresh, np.argmax(pred, axis=1), n_closed)
    
#         a = np.sum(np.where(label == pred, 1, 0))/pred.shape[0]*100

#         return a, label, pred

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



# datapath = "/home/sec_user/thilini/Anchor_loss/Tor/dataset/DF_dataset/dataset/"

dataXr, y_train, _, _, _, _ = LoadDataNoDefCW()
# dataXr = np.load(datapath+"X_train.npy")
# y_train = np.load(datapath+"y_train.npy")
dataXr, y_train = get_small_set(dataXr, y_train, 200)
dataXr = dataXr[:, 0:1500, np.newaxis]


model = load_model("/media/SATA_1/thilini_open_extra/final_codes/OpenMax/AWF/closed/AWF_closed_keras.h5py")
pred = model.predict(dataXr)


model_path =  "/media/SATA_1/thilini_open_extra/final_codes/OpenMax/AWF/closed/AWF_closed_keras.h5py"
sub_model = build(n_closed, model_path)
prenultimate_layer_out = sub_model.predict(dataXr)


openmax_ob = Openmax(alpharank=1, tailsize=10, decision_dist_fn='euclidean')
openmax_ob.update_class_stats(prenultimate_layer_out, pred, keras.utils.to_categorical(y_train))
print('updated class data')


_, _, dataXr, y_train, _, _ = LoadDataNoDefCW()
# dataXr = np.load(datapath+"X_valid.npy")
# y_train = np.load(datapath+"y_valid.npy")
dataXr, y_train = get_small_set(dataXr, y_train, 100)
dataXr = dataXr[:, 0:1500, np.newaxis]

prenultimate_layer_out = sub_model.predict(dataXr)
open_prob = openmax_ob.predict_prob_open(prenultimate_layer_out)


min_list = getMinList(open_prob, y_train)

print('minlist created')


_, _, _, _, xtest, ytest = LoadDataNoDefCW()
# xtest = np.load(datapath+"X_test.npy")
xtest = xtest[:, 0:1500, np.newaxis]
# ytest = np.load(datapath+"y_test.npy")

logits_test = sub_model.predict(xtest)
open_prob_c = openmax_ob.predict_prob_open(logits_test)

del logits_test, xtest

# acc, pred1, label1 = accuracy(open_prob_c, ytest, min_list, 0)
# # rese[i] = accuracy(open_prob, ytest, 0.0, 0)
# print('Accuracy closed classwise:{} \n'.format(acc))

# del acc, pred1, label1

datapath = '/media/SATA_1/thilini_open_extra/final_datasets/AWF/'
openx = np.load(datapath+'X_open.npy')
openx = openx[:, 0:1500, np.newaxis]
openy = np.ones((openx.shape[0],))*n_closed

logits_test = sub_model.predict( openx)
open_prob_o = openmax_ob.predict_prob_open(logits_test)

del logits_test, openx

closed_acc, open_acc, f1 = accuracy(ytest, openy, open_prob_c,  open_prob_o, min_list, n_closed)
print('closed acc:{}'.format(closed_acc))
print('open acc:{}'.format(open_acc))
print('f_score:{}'.format(f1))












