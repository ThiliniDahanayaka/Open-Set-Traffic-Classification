import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
#from operator import add

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from utility import LoadDataIot
from Model_cnn import DCNet
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import preprocessing as sp
import pandas as pd 
from tensorflow_addons.optimizers import AdamW
from openmax import Openmax

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train_Rep).batch(1).take(100):
    yield [input_value]
    
def Micro_F1(matrix):
  epsilon=1e-8
  TP=0
  FP=0
  TN=0
  
  for k in range(NB_CLASSES):
    TP+=matrix[k][k]
    FP+=(np.sum(Matrix,axis=0)[k]-matrix[k][k])
    TN+=(np.sum(Matrix,axis=1)[k]-matrix[k][k])
    
  Micro_Prec=TP/(TP+FP)
  Micro_Rec=TP/(TP+TN)
  print("Micro Precision: ", Micro_Prec)
  print("Micro Recall: ", Micro_Rec)
  Micro_F1=2*Micro_Prec*Micro_Rec/(Micro_Rec+Micro_Prec+epsilon)
  
  return Micro_F1


def New_F1_Score(Matrix):
  Column_sum=np.sum(Matrix,axis=0)
  Raw_sum=np.sum(Matrix,axis=1)
  
  Precision_Differences=[]
  Recall_Differences=[]
  for i in range(NB_CLASSES):
    Precision_Differences.append(np.abs(2*Matrix[i][i]-Column_sum[i]))
    Recall_Differences.append(np.abs(2*Matrix[i][i]-Raw_sum[i]))
  
  Precision_Differences=np.array(Precision_Differences)
  Precision_Differences_Per=Precision_Differences/np.sum(Precision_Differences)
  Recall_Differences=np.array(Recall_Differences)
  Recall_Differences_Per=Recall_Differences/np.sum(Recall_Differences)
  
  #print('Precision_Differences_Per',Precision_Differences_Per)
  #print('Recall_Differences_Per',Recall_Differences_Per)
  
  Precisions=np.zeros(NB_CLASSES)
  Recalls=np.zeros(NB_CLASSES)
  
  epsilon=1e-8
  
  for k in range(len(Precisions)):
    Precisions[k]=(Matrix[k][k]/np.sum(Matrix,axis=0)[k])
  Precision=np.sum(np.array(Precisions)*Precision_Differences_Per)
  
  for k in range(len(Recalls)):
    Recalls[k]=(Matrix[k][k]/np.sum(Matrix,axis=1)[k])  #*Recall_Differences_Per[k]
  Recall=np.sum(np.array(Recalls)*Recall_Differences_Per)
  
  print('Precision: ',Precision)
  print('Recall: ',Recall)
    
  
  F1_Score=2*Precision*Recall/(Precision+Recall+epsilon)
  return F1_Score

def Macro_F1(Matrix):
  Precisions=np.zeros(NB_CLASSES)
  Recalls=np.zeros(NB_CLASSES)
  
  epsilon=1e-8
  
  for k in range(len(Precisions)):
    Precisions[k]=Matrix[k][k]/np.sum(Matrix,axis=0)[k]
  #print(Precisions)
    
  Precision=np.average(Precisions)
  print("Precision:",Precision)
    
  for k in range(len(Recalls)):
    Recalls[k]=Matrix[k][k]/np.sum(Matrix,axis=1)[k]
   
  Recall=np.average(Recalls)
  print("Recall:",Recall)

  F1_Score=2*Precision*Recall/(Precision+Recall+epsilon)
  return F1_Score

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
    
  

random.seed(0)


# Training the DF model
NB_EPOCH = 1000   # Number of training epoch
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 64 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 500 # Packet sequence length

NB_CLASSES = 4 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()
# Please refer to the dataset format in readme
dataset_dir = "/home/Yasod/DC/dataset/"


# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
#X_open = X_open.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('int8')


X_train=X_train[:-400]
y_train=y_train[:-400]

X_train = X_train[:, :,np.newaxis]
X__train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]


X_train_Rep,y_train_Rep=shuffle(X__train, y_train)
np.save('./Temp/X_train_Rep.npy',X_train_Rep,allow_pickle=True)
np.save('./Temp/y_train_Rep.npy',y_train_Rep,allow_pickle=True)

#X_train_Rep=np.load('./Temp/X_train_Rep.npy',allow_pickle=True)
#y_train_Rep=np.load('./Temp/y_train_Rep.npy',allow_pickle=True)



#y_train = to_categorical(y_train, NB_CLASSES)
#y_valid = to_categorical(y_valid, NB_CLASSES)
#y_test_ = to_categorical(y_test, NB_CLASSES)
#y_open_ = to_categorical(y_open, NB_CLASSES)

#y_test_=np.argmax(y_test_, axis=1)


Mean_0=np.array([0]*NB_CLASSES)
Mean_1=np.array([0]*NB_CLASSES)
Mean_2=np.array([0]*NB_CLASSES)
Mean_3=np.array([0]*NB_CLASSES)
Mean_4=np.array([0]*NB_CLASSES)
Mean_5=np.array([0]*NB_CLASSES)

model=load_model('DC_without_softmax.hdf5')  # This is the model we train on the DC dataset.
                                             # The last layer should be the phenultimate layer.
tflite_model_predictions = model.predict(X_train)
#pred=np.argmax(tflite_model_predictions,axis=1)

openmax_ob = Openmax(alpharank=1, tailsize=10, decision_dist_fn='euclidean')
openmax_ob.update_class_stats(tflite_model_predictions, tflite_model_predictions, to_categorical(y_train))
print('updated class data')

with open('/home/Yasod/DC/Openmax_trained', 'wb') as file_pi:
        pickle.dump(openmax_ob, file_pi)


for i in range(len(tflite_model_predictions)):
  if (y_train[i]==0 and np.argmax(tflite_model_predictions[i])==y_train[i]):
    Mean_0=np.sum([Mean_0,tflite_model_predictions[i]], axis=0)
    
  elif (y_train[i]==1 and np.argmax(tflite_model_predictions[i])==y_train[i]):
    Mean_1=np.sum([Mean_1,tflite_model_predictions[i]], axis=0)
    
  elif (y_train[i]==2 and np.argmax(tflite_model_predictions[i])==y_train[i]):
    Mean_2=np.sum([Mean_2,tflite_model_predictions[i]], axis=0)
    
  elif (y_train[i]==3 and np.argmax(tflite_model_predictions[i])==y_train[i]):
    Mean_3=np.sum([Mean_3,tflite_model_predictions[i]], axis=0)
    
  elif (y_train[i]==4 and np.argmax(tflite_model_predictions[i])==y_train[i]):
    Mean_4=np.sum([Mean_4,tflite_model_predictions[i]], axis=0)
    
  elif (y_train[i]==5 and np.argmax(tflite_model_predictions[i])==y_train[i]):
    Mean_5=np.sum([Mean_5,tflite_model_predictions[i]], axis=0)
    
Mean_0=Mean_0/list(y_train).count(0)
Mean_1=Mean_1/list(y_train).count(1)
Mean_2=Mean_2/list(y_train).count(2)
Mean_3=Mean_3/list(y_train).count(3)

Mean_vectors=np.array([Mean_0,Mean_1,Mean_2,Mean_3])
np.save('./Temp/Mean_vectors.npy', Mean_vectors, allow_pickle=True)

tflite_model_predictions = model.predict(X_valid)
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)

Dist_0=[]
Dist_1=[]
Dist_2=[]
Dist_3=[]
Dist_4=[]
Dist_5=[]

def normal_TH():
  for i in range(len(tflite_model_predictions)):
    if (y_valid[i]==0 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      dist = np.linalg.norm(Mean_0-tflite_model_predictions[i])
      Dist_0.append(dist)
      
    elif (y_valid[i]==1 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      dist = np.linalg.norm(Mean_1-tflite_model_predictions[i])
      Dist_1.append(dist)
      
    elif (y_valid[i]==2 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      dist = np.linalg.norm(Mean_2-tflite_model_predictions[i])
      Dist_2.append(dist)
      
    elif (y_valid[i]==3 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      dist = np.linalg.norm(Mean_3-tflite_model_predictions[i])
      Dist_3.append(dist)
      
    elif (y_valid[i]==4 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      dist = np.linalg.norm(Mean_4-tflite_model_predictions[i])
      Dist_4.append(dist)
      
    elif (y_valid[i]==5 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      dist = np.linalg.norm(Mean_5-tflite_model_predictions[i])
      Dist_5.append(dist)
    
def open_max_Th(open_prob):
  for i in range(len(tflite_model_predictions)):
    if (y_valid[i]==0 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      Dist_0.append(open_prob[i][int(y_valid[i])])
      
    elif (y_valid[i]==1 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      Dist_1.append(open_prob[i][int(y_valid[i])])
      
    elif (y_valid[i]==2 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      Dist_2.append(open_prob[i][int(y_valid[i])])
      
    elif (y_valid[i]==3 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      Dist_3.append(open_prob[i][int(y_valid[i])])
      
    elif (y_valid[i]==4 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      Dist_4.append(open_prob[i][int(y_valid[i])])
      
    elif (y_valid[i]==5 and np.argmax(tflite_model_predictions[i])==y_valid[i]):
      Dist_5.append(open_prob[i][int(y_valid[i])])

open_max_Th(open_prob)

Dist_0.sort()
Dist_1.sort()
Dist_2.sort()
Dist_3.sort()

percentile=0.10
Th_0=Dist_0[int(len(Dist_0)*percentile)]
Th_1=Dist_1[int(len(Dist_1)*percentile)]
Th_2=Dist_2[int(len(Dist_2)*percentile)]
Th_3=Dist_3[int(len(Dist_3)*percentile)]

Threasholds=np.array([Th_0,Th_1,Th_2,Th_3])
np.save('./Temp/Threasholds.npy',Threasholds)


print("Thresholds are calculated")


print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()
# Please refer to the dataset format in readme
dataset_dir = "/home/Yasod/DC/dataset/"
X_open = np.load(dataset_dir+'X_open.npy')
y__open = np.load(dataset_dir+'y_open.npy')
y_open=np.array([NB_CLASSES]*len(y__open))
#y_test=np.concatenate((y_test, y_open), axis=0)
#X_test=np.concatenate((X_test, X_open), axis=0)


print("#####################################")


# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_open = X_open.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('int8')
y_open = y_open.astype('int8')
# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)


Threasholds=np.load('./Temp/Threasholds.npy',allow_pickle=True)
Mean_Vectors=np.load('./Temp/Mean_vectors.npy')
print(Mean_Vectors.shape)
Mean_0=Mean_Vectors[0]
Mean_1=Mean_Vectors[1]
Mean_2=Mean_Vectors[2]
Mean_3=Mean_Vectors[3]

#X_test = X_test[:, :,np.newaxis]
X_open = X_open[:, :,np.newaxis]

model=load_model('DC_without_softmax.hdf5')
with open('/home/Yasod/DC/Openmax_trained', 'rb') as f:
    openmax_ob = pickle.load(f)


tflite_model_predictions = model.predict(X_test)
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)

prediction_classes=[]
for i in range(len(open_prob)):
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes.append(d)
    elif open_prob[i][d]<Threasholds[d]:
      prediction_classes.append(NB_CLASSES)
      
    else:
      prediction_classes.append(d)
      
  
    
  
prediction_classes=np.array(prediction_classes)  

acc_Close = accuracy_score(prediction_classes, y_test)
print('Test accuracy Normal model_Closed_set :', acc_Close)


# //////////////////////////////////////////////////////////////////////////////////////////////

tflite_model_predictions_open = model.predict(X_open)
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions_open)

prediction_classes_open=[]
for i in range(len(open_prob)):
  
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes_open.append(d)
    elif open_prob[i][d]<Threasholds[d]:
      prediction_classes_open.append(NB_CLASSES)
      
    else:
      prediction_classes_open.append(d)
      
  
prediction_classes_open=np.array(prediction_classes_open)  

acc_Open = accuracy_score(prediction_classes_open, y_open)
print('Test accuracy Normal model_Open_set :', acc_Open)



Matrix=[]
for i in range(NB_CLASSES+1):
  Matrix.append(np.zeros(NB_CLASSES+1))
  
  
for i in range(len(y_test)):
  Matrix[y_test[i]][prediction_classes[i]]+=1
  
for i in range(len(y_open)):
  Matrix[y_open[i]][prediction_classes_open[i]]+=1
  
print(Matrix)
print()

F1_Score=New_F1_Score(Matrix)

print("Average F1_Score: ", F1_Score)

print()
print("Micro")
F1_Score=Micro_F1(Matrix)
print("Average F1_Score: ", F1_Score)

print()
print("Macro")
F1_Score=Macro_F1(Matrix)
print("Average F1_Score: ", F1_Score)
print()


