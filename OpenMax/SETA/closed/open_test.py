import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

from keras import backend as K
from keras.models import load_model
from utility import LoadDataNoDefCW
from Model_NoDef import DFNet
import random
from keras.utils import np_utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# from conf_mat_plot import plot_confusion_matrice
import numpy as np
import argparse
import sklearn.metrics
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

NB_CLASSES = 8 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)

parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--trial', default = 1, type = int, help='Trial number, 0-4 provided')
args = parser.parse_args()
trial = str(args.trial)
dataset="SETA"

def accuracy(Inliers_true, other_true, Inliers_score_raw,  other_score_raw, num_classes):

    Inliers_pred = np.argmax(Inliers_score_raw, axis=1)
    other_pred = np.argmax(other_score_raw, axis=1)

    Inliers_score = np.max(Inliers_score_raw, axis=1)
    other_score = np.max(other_score_raw, axis=1)
    
    start = np.min(Inliers_score)
    end = np.max(Inliers_score)
    # print(Inliers_score)
    gap = 0.002 #(end- start)/200000 # precision:200000

    print(start, end, gap)
    best_acc_ave= 0.0
    best_threshold = start
    accuracy_thresh = 90.0 
    accuracy_range = np.arange(start, end, gap)
    if len(accuracy_range)==0:
        accuracy_range = np.array([start, start-0.002])
    for i, delta in enumerate(accuracy_range):
        # samples with prediction probabilities less than thresh are labeld as open
        Inliers_label = np.where(Inliers_score>=delta, Inliers_pred, num_classes)
        # Calculate accuracy 
        a = np.sum(np.where(Inliers_true == Inliers_label, 1, 0))/Inliers_label.shape[0]*100       
        Outliers_label = np.where(other_score>=delta, other_pred, num_classes)       
        b = np.sum(np.where(other_true == Outliers_label, 1, 0))/Outliers_label.shape[0]*100  
        # print('i:{}, delta:{}, close:{}, open:{}'.format(i, delta, a, b))

        if i==0 and a<accuracy_thresh:
            print('Closed set accuracy did not reach ', accuracy_thresh)
            return a, b, delta   

        # if (a+b)/2 >best_acc_ave and a>=90.:
        if a<accuracy_thresh and i>0:
            print('ideal')
            delta = accuracy_range[i-1]
            best_threshold = delta

            Inliers_label = np.where(Inliers_score>=delta, Inliers_pred, num_classes)
            # Calculate accuracy 
            best_acc_inlier = np.sum(np.where(Inliers_true == Inliers_label, 1, 0))/Inliers_label.shape[0]*100       
            Outliers_label = np.where(other_score>=delta, other_pred, num_classes)       
            best_acc_outlier = np.sum(np.where(other_true == Outliers_label, 1, 0))/Outliers_label.shape[0]*100  
            
            best_acc_ave = (best_acc_inlier+best_acc_outlier)/2
            # print('\ti:{}, delta:{}, close:{}, open:{}'.format(i, delta, a, b))
            
            return best_acc_inlier, best_acc_outlier, best_threshold

        if i==len(accuracy_range)-1:
            print('Closed set accuracy did not fall below 90')
            return a, b, delta
   

        
    

def auroc(inData, outData, title='test', trial_num=1):
    # print(inData.shape, outData.shape)
    inDataMin = np.max(inData, 1)
    outDataMin = np.max(outData, 1)

    Y1 = outDataMin
    X1 = inDataMin
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    # end = np.min(X1)
    # start = np.max(X1)
    gap = (end- start)/200000

    print(start, end, gap)

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        # f1.write("{}\n".format(tpr))
        # f2.write("{}\n".format(fpr))
        aurocBase += (-fpr+fprTemp)*tpr
        # print('delta:{}, tpr:{}, fpr:{}, fprTemp:{}, aurocBase:{}'.format(delta,    tpr, fpr, fprTemp, aurocBase))
        fprTemp = fpr

    return aurocBase

def get_f1(Inliers_true, other_true, Inliers_score_raw,  other_score_raw, delta, num_classes):
    '''
    delta: threshold value to reject open samples calculated from method 'accuracy'
    '''

    # print(Inliers_true.shape, other_true.shape)
    Inliers_pred = np.argmax(Inliers_score_raw, axis=1)
    other_pred = np.argmax(other_score_raw, axis=1)

    Inliers_score = np.max(Inliers_score_raw, axis=1)
    other_score = np.max(other_score_raw, axis=1)

    Inliers_label = np.where(Inliers_score>=delta, Inliers_pred, num_classes)
    Outliers_label = np.where(other_score>=delta, other_pred, num_classes)

    prediction = np.append(Inliers_label, Outliers_label, axis=0)
    labels = np.append(Inliers_true, other_true, axis=0)

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

        P = P+(tp/(tp+fp))
        R = R+(tp/(tp+fn))



    P = P/num_classes
    R = R/num_classes

    F = 2*P*R/(P+R)

    return F


# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
_, _, X_valid, y_valid, X_test, y_test, X_open, y_open = LoadDataNoDefCW('SETA', NB_CLASSES, trial_num=trial)



filepath = '/media/SATA_1/thilini_open_extra/final_codes/OpenMax/SETA/closed/models/seta_closed_trial_'+trial+'.hdf5'
model = load_model(filepath)
# thresh = 0.8

score = model.evaluate(X_test, np_utils.to_categorical(y_test, NB_CLASSES))
print(score)

out_c = model.predict(X_valid)
# thresh = find_thresh(out, y_valid, NB_CLASSES)
best_acc_inlier, _, best_threshold = accuracy(y_valid, y_valid, out_c,  out_c, NB_CLASSES)
print('validation acc:{}, thrresh:{}'.format(best_acc_inlier, best_threshold))


out_c = model.predict(X_test)
out_o = model.predict(X_open)
best_acc_inlier, best_acc_outlier, _ = accuracy(y_test, y_open, out_c,  out_o, NB_CLASSES)

f1 = get_f1(y_test, y_open, out_c,  out_o, best_threshold, NB_CLASSES)

auroc = auroc(out_c,  out_o, title='test', trial_num=1)

print('best thresh:{}'.format(best_threshold))
print('closed acc:{}'.format(best_acc_inlier))
print('open acc:{}'.format(best_acc_outlier))
print('auroc:{}'.format(auroc*100))
print('f_score:{}'.format(f1))