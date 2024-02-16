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
  for input_value in tf.data.Dataset.from_tensor_slices(X_train_Rep).batch(1).take(1000):
    # Model has only one input so each data point has one element.
    yield [input_value]
    
def representative_data_gen_Open():
  for input_value in tf.data.Dataset.from_tensor_slices(X_open).batch(1).take(1000):
    # Model has only one input so each data point has one element.
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



random.seed(0)


# Training the DF model
NB_EPOCH = 500   # Number of training epoch
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
X_open = np.load(dataset_dir+'X_open.npy')
#X_open=X_open[:173]
#y_open = np.load(dataset_dir+'y_open.npy')
#print("y_open_shape: ",y_open.shape)
#y_open=y_open[:173]
#y_open=np.array([5]*173)

#print(X_test[0])

print("#####################################")

print("Xtrain min",np.min(X_train))
print("Xtrain max",np.max(X_train))

interval_min = -128
interval_max = 127
X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train)) * (interval_max - interval_min) + interval_min
X_valid = (X_valid - np.min(X_valid)) / (np.max(X_valid) - np.min(X_valid)) * (interval_max - interval_min) + interval_min
#X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test)) * (interval_max - interval_min) + interval_min
#X_open = (X_open - np.min(X_open)) / (np.max(X_open) - np.min(X_open)) * (interval_max - interval_min) + interval_min
#print(X_train[0])
#print(X_valid[0])


# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_open = X_open.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('int8')

X_valid=np.concatenate((X_valid,X_train[-400:]),axis=0)
y_valid=np.concatenate((y_valid,y_train[-400:]),axis=0)

X_train=X_train[:-400]
y_train=y_train[:-400]

print(X_valid.shape)
print(y_valid.shape)
#y_open = y_open.astype('int8')
# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X__train = X_train[:, :,np.newaxis]
X_open = X_open[:, :,np.newaxis]


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




model=load_model('DC_without_softmax_s5.hdf5')
print(X_test.shape)


TF_LITE_MODEL_FILE_NAME = "tf_lite_fullint_softmax.tflite"

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]       # COnvert to 16 bit
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
#tflite_model.save('DC_without_norm.hdf5')

tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name, "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

interpreter.resize_tensor_input(input_details[0]['index'], (len(X_train), 500, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (len(X_train), NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])
    



input_scale,input_zero_point  = input_details[0]["quantization"]
print("################",input_scale)
print("################",input_zero_point)
for i in range(len(X_test)):
  X_train[i] = X_train[i] / input_scale + input_zero_point
  
print("Xtrain ",X_train[:25]) 
#print(X_test[0])
X_train = X_train[:, :,np.newaxis]
X_train = X_train.astype('int8')

#X_open = X_open[:, :,np.newaxis]
#X_open = X_open.astype('int8')
#interpreter.set_tensor(input_details[0]['index'], X_test)
#interpreter.invoke()
#tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
#print("Prediction results shape:", tflite_model_predictions.shape)
#print("y_test:", y_test.shape)
#print()

interpreter.set_tensor(input_details[0]['index'], X_train)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])

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


interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

interpreter.resize_tensor_input(input_details[0]['index'], (len(X_valid), 500, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (len(X_valid), NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



input_scale,input_zero_point  = input_details[0]["quantization"]

for i in range(len(X_valid)):
  X_valid[i] = X_valid[i] / input_scale + input_zero_point

print('***',X_valid[:10])
X_valid = X_valid[:, :,np.newaxis]
X_valid = X_valid.astype('int8')
#X_open = X_open[:, :,np.newaxis]
#X_open = X_open.astype('int8')
#interpreter.set_tensor(input_details[0]['index'], X_test)
#interpreter.invoke()
#tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
#print("Prediction results shape:", tflite_model_predictions.shape)
#print("y_test:", y_test.shape)
#print()

interpreter.set_tensor(input_details[0]['index'], X_valid)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])

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
#Dist_4.sort()
#Dist_5.sort()
print(len(Dist_0))
print(len(Dist_1))
print(len(Dist_2))
print(len(Dist_3))
#print(len(Dist_4))

percentile=0.1
Th_0=Dist_0[int(len(Dist_0)*percentile)]
Th_1=Dist_1[int(len(Dist_1)*percentile)]
Th_2=Dist_2[int(len(Dist_2)*percentile)]
Th_3=Dist_3[int(len(Dist_3)*percentile)]
#Th_4=Dist_4[int(len(Dist_4)*percentile)-1]
#Th_5=Dist_5[int(len(Dist_5)*percentile)]

Threasholds=np.array([Th_0,Th_1,Th_2,Th_3])
print(Threasholds)
np.save('./Temp/Threasholds.npy',Threasholds)


print("Done")




TF_LITE_MODEL_FILE = "tf_lite_Open_set.tflite"

converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]       # COnvert to 16 bit
converter2.representative_dataset = representative_data_gen_Open
converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter2.inference_input_type = tf.int8  # or tf.uint8
converter2.inference_output_type = tf.int8

tflite_model = converter2.convert()
#tflite_model.save('DC_without_norm.hdf5')

tflite_model_name = TF_LITE_MODEL_FILE
open(tflite_model_name, "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE)

input_details = interpreter.get_input_details()
input_scale_open,input_zero_point_open  = input_details[0]["quantization"]
np.save('./Temp/Input_Convertion_Opne.npy',np.array([input_scale_open,input_zero_point_open]))
    

random.seed(0)


# Training the DF model
NB_EPOCH = 500   # Number of training epoch
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 64 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 500 # Packet sequence length
OPTIMIZER = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # Optimizer

NB_CLASSES = 4 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()
# Please refer to the dataset format in readme
dataset_dir = "/home/Yasod/DC/dataset/"
X_open = np.load(dataset_dir+'X_open.npy')
#X_open=X_open[:173]
y__open = np.load(dataset_dir+'y_open.npy')
print("y_open_shape: ",y__open.shape)
#y_open=y_open[:173]
y_open=np.array([NB_CLASSES]*len(y__open))
#y_test=np.concatenate((y_test, y_open), axis=0)
#X_test=np.concatenate((X_test, X_open), axis=0)
print("X_test_shape: ",X_test.shape)
print("y_test_shape: ",y_test.shape)

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

X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]


print()



y_train = to_categorical(y_train, NB_CLASSES)
y_valid = to_categorical(y_valid, NB_CLASSES)
y_test_ = to_categorical(y_test, NB_CLASSES)
#y_open_ = to_categorical(y_open, NB_CLASSES)

y_test_=np.argmax(y_test_, axis=1)


interpreter = tf.lite.Interpreter(model_path = 'tf_lite_fullint_softmax.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

interpreter.resize_tensor_input(input_details[0]['index'], (len(X_test), 500, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (len(X_test), NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])
    
Threasholds=np.load('./Temp/Threasholds.npy',allow_pickle=True)
Mean_Vectors=np.load('./Temp/Mean_vectors.npy',allow_pickle=True)
[input_scale_open,input_zero_point_open]=np.load('./Temp/Input_Convertion_Opne.npy',allow_pickle=True)

with open('/home/Yasod/DC/Openmax_trained', 'rb') as f:
    openmax_ob = pickle.load(f)

print(Mean_Vectors.shape)
Mean_0=Mean_Vectors[0]
Mean_1=Mean_Vectors[1]
Mean_2=Mean_Vectors[2]
Mean_3=Mean_Vectors[3]
#Mean_4=Mean_Vectors[4]


input_scale,input_zero_point  = input_scale_open,input_zero_point_open
print("################",input_scale)
print("################",input_zero_point)
for i in range(len(X_test)):
  X_test[i] = X_test[i] / input_scale + input_zero_point
for i in range(len(X_open)):
  X_open[i] = X_open[i] / input_scale + input_zero_point 

#print(X_test[0])
X_test = X_test[:, :,np.newaxis]
X_test = X_test.astype('int8')
X_open = X_open[:, :,np.newaxis]
X_open = X_open.astype('int8')

interpreter.set_tensor(input_details[0]['index'], X_test)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])

open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)
#print("######### Closed set ##############")
#print(tflite_model_predictions[:5])

prediction_classes=[]
for i in range(len(tflite_model_predictions)):
  
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes.append(d)
    elif open_prob[i][d]<Threasholds[d]:
      prediction_classes.append(NB_CLASSES)
      
    else:
      prediction_classes.append(d)
      
prediction_classes=np.array(prediction_classes) 


acc_Close = accuracy_score(prediction_classes, y_test)
print('Test accuracy TFLITE model_Closed_set :', acc_Close)


# //////////////////////////////////////////////////////////////////////////////////////////////

interpreter = tf.lite.Interpreter(model_path = 'tf_lite_fullint_softmax.tflite')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

interpreter.resize_tensor_input(input_details[0]['index'], (len(X_open), 500, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (len(X_open), NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], X_open)
interpreter.invoke()
tflite_model_predictions_open = interpreter.get_tensor(output_details[0]['index'])
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions_open)

prediction_classes_open=[]
for i in range(len(tflite_model_predictions_open)):
  
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes_open.append(d)
    elif open_prob[i][d]<Threasholds[d]:
      prediction_classes_open.append(NB_CLASSES)
      
    else:
      prediction_classes_open.append(d)
      
  
prediction_classes_open=np.array(prediction_classes_open)  

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