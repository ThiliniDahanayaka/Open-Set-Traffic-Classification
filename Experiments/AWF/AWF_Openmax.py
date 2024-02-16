import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from utility import LoadDataNoDefCW
from Model_NoDef import DFNet
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import preprocessing as sp
import pandas as pd 
from openmax import Openmax
import pandas as pd 


def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train_Rep).batch(1).take(100):
    yield [input_value]

def Macro_F1(Matrix):
  Precisions=np.zeros(NB_CLASSES)
  Recalls=np.zeros(NB_CLASSES)
  
  epsilon=1e-8
  
  for k in range(len(Precisions)):
    Precisions[k]=Matrix[k][k]/np.sum(Matrix,axis=0)[k]
    
  Precision=np.average(Precisions)
    
  for k in range(len(Recalls)):
    Recalls[k]=Matrix[k][k]/np.sum(Matrix,axis=1)[k]
   
  Recall=np.average(Recalls)

  F1_Score=2*Precision*Recall/(Precision+Recall+epsilon)
  return F1_Score
    
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

random.seed(0)


# Training the DF model
NB_EPOCH = 500   # Number of training epoch
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 64 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 1500 # Packet sequence length
OPTIMIZER = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # Optimizer

NB_CLASSES = 200 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataNoDefCW()

dataset_dir = "/home/ubuntu/Yasod/AWF/dataset/"



# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
#X_open = X_open.astype('float32')
y_train = y_train.astype('int16')
y_valid = y_valid.astype('int16')
y_test = y_test.astype('int16')
#y_open = y_open.astype('int8')
# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]

txt_O = "Mean_{Class1:.0f}"
Means={}
for i in range(NB_CLASSES):
  Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)
  


model=load_model('AWF_without_softmax.hdf5')
tflite_model_predictions = model.predict(X_train)

openmax_ob = Openmax(alpharank=1, tailsize=10, decision_dist_fn='euclidean')
openmax_ob.update_class_stats(tflite_model_predictions, tflite_model_predictions, to_categorical(y_train))
print('updated class data')

with open('/home/Yasod/Yasod/AWF/Openmax_trained', 'wb') as file_pi:
        pickle.dump(openmax_ob, file_pi)

print('Shape: ',tflite_model_predictions.shape)
count=[0]*NB_CLASSES

for i in range(len(tflite_model_predictions)):
  k=np.argmax(tflite_model_predictions[i])
  if (np.argmax(tflite_model_predictions[i])==y_train[i]):
    Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])]+tflite_model_predictions[i]
    count[y_train[i]]+=1
print("Counts: ",count)

Mean_Vectors=[]   
for i in range(NB_CLASSES):
  Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
  Mean_Vectors.append(Means[txt_O.format(Class1=i)])

Mean_vectors=np.array(Mean_Vectors)
np.save('./Temp/Mean_vectors.npy', Mean_vectors, allow_pickle=True)

tflite_model_predictions=model.predict(X_valid)
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)

txt_1 = "Dist_{Class1:.0f}"
Distances={}
for i in range(NB_CLASSES):
  Distances[txt_1.format(Class1=i)]=[]
  
  
for i in range(len(open_prob)):
  if (y_valid[i]==np.argmax(tflite_model_predictions[i])):
    Distances[txt_1.format(Class1=y_valid[i])].append(open_prob[i][int(y_valid[i])])
   
TH=[0]*NB_CLASSES  
for j in range(NB_CLASSES):
  Distances[txt_1.format(Class1=j)].sort()
  Dist=Distances[txt_1.format(Class1=j)]
  TH[j]=Dist[int(len(Dist)*0.1)]  



Threasholds=np.array(TH)
np.save('./Temp/Threasholds.npy',Threasholds)


print("Thresholds are Caluculated")


# /////////////////////////////////////////////////////////////////////////

X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataNoDefCW()
del X_train,y_train,X_valid,y_valid

gc.collect()
# Please refer to the dataset format in readme
dataset_dir = "/home/ubuntu/Yasod/AWF/dataset/"

Data = np.load("/home/Yasod/Yasod/AWF/dataset/openset.npz",allow_pickle=True)
X_open=np.array(Data['data'])
X_open=X_open[:,:1500]
y_open = np.array([NB_CLASSES]*len(X_open))

X_test,y_test=shuffle(X_test, y_test)

X_test = X_test.astype('float32')
X_open = X_open.astype('float32')

y_test = y_test.astype('int16')


model=load_model('AWF_without_softmax.hdf5')
Mean_Vectors=np.load('./Temp/Mean_vectors.npy')
Threasholds=np.load('./Temp/Threasholds.npy')

with open('/home/Yasod/Yasod/AWF/Openmax_trained', 'rb') as f:
    openmax_ob = pickle.load(f)

X_test = X_test[:, :,np.newaxis]

X_open = X_open[:, :,np.newaxis]


prediction_classes=[]
tflite_model_predictions=model.predict(X_test)
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)

for i in range(len(tflite_model_predictions)):
  
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes.append(d)
    elif open_prob[i][d]<Threasholds[d]:
      prediction_classes.append(NB_CLASSES)
      
    else:
      prediction_classes.append(d)
      

    

prediction_classes_open=[]
tflite_model_predictions_open=model.predict(X_open)
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions_open)


for i in range(len(tflite_model_predictions_open)):
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes_open.append(d)
    elif open_prob[i][d]<Threasholds[d]:
      prediction_classes_open.append(NB_CLASSES)
      
    else:
      prediction_classes_open.append(d)
    
    
acc_Close = accuracy_score(prediction_classes, y_test)
print('Test accuracy TFLITE model_Closed_set :', acc_Close)

acc_Open = accuracy_score(prediction_classes_open, y_open)
print('Test accuracy TFLITE model_Open_set :', acc_Open)


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
print('Test accuracy TFLITE model_Closed_set :', acc_Close)
print('Test accuracy TFLITE model_Open_set :', acc_Open)
F1_Score=Micro_F1(Matrix)
print("Average F1_Score: ", F1_Score)

print()
print("Macro")
print('Test accuracy TFLITE model_Closed_set :', acc_Close)
print('Test accuracy TFLITE model_Open_set :', acc_Open)
F1_Score=Macro_F1(Matrix)
print("Average F1_Score: ", F1_Score)


