import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

print("#####################################")



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
X__train = X_train[:, :,np.newaxis]
#X_valid = X_valid[:, :,np.newaxis]

X_train_Rep,y_train_Rep=shuffle(X__train, y_train)

txt_O = "Mean_{Class1:.0f}"
Means={}
for i in range(NB_CLASSES):
  Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)
  


model=load_model('AWF_without_softmax.hdf5')
print(X_test.shape)


TF_LITE_MODEL_FILE_NAME = "tf_lite_fullint_softmax_openmax.tflite"

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

interpreter.resize_tensor_input(input_details[0]['index'], (5000, LENGTH, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (5000, NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()




input_scale,input_zero_point  = input_details[0]["quantization"]
for i in range(len(X_test)):
  X_train[i] = X_train[i] / input_scale + input_zero_point
  
#print(X_test[0])
X_train = X_train[:, :,np.newaxis]
X_train = X_train.astype('int8')
X_open = X_open[:, :,np.newaxis]
X_open = X_open.astype('int8')
interpreter.set_tensor(input_details[0]['index'], X_test)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])

tflite_model_predictions = []
for i in range(4):
  interpreter.set_tensor(input_details[0]['index'], X_train[5000*i:5000*(i+1)])
  interpreter.invoke()
  Mod_Prediction=interpreter.get_tensor(output_details[0]['index'])
  if i==0:
    tflite_model_predictions=Mod_Prediction
  else:
    tflite_model_predictions=np.concatenate((tflite_model_predictions,Mod_Prediction),axis=0)
  print(i,"Done")

#print('Shape: ',tflite_model_predictions.shape)
count=[0]*NB_CLASSES
np.save('./Temp/X_train_quantized.npy',tflite_model_predictions)

#tflite_model_predictions=np.load('X_train_quantized.npy')
openmax_ob = Openmax(alpharank=1, tailsize=10, decision_dist_fn='euclidean')
openmax_ob.update_class_stats(tflite_model_predictions, tflite_model_predictions, to_categorical(y_train[:len(tflite_model_predictions)]))
print('updated class data')

with open('/home/Yasod/Yasod/AWF/Openmax_trained', 'wb') as file_pi:
        pickle.dump(openmax_ob, file_pi)


for i in range(len(tflite_model_predictions)):
  k=np.argmax(tflite_model_predictions[i])
  if (np.argmax(tflite_model_predictions[i])==y_train[i]):
    Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])]+tflite_model_predictions[i]
    count[y_train[i]]+=1


Mean_Vectors=[]   
for i in range(NB_CLASSES):
  Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
  Mean_Vectors.append(Means[txt_O.format(Class1=i)])

Mean_Vectors=np.load('Mean_vectors.npy', allow_pickle=True)
np.save('Mean_vectors.npy', Mean_vectors, allow_pickle=True)


interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

interpreter.resize_tensor_input(input_details[0]['index'], (5000, 1500, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (5000, NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
    


input_scale,input_zero_point  = input_details[0]["quantization"]
for i in range(len(X_valid)):
  X_valid[i] = X_valid[i] / input_scale + input_zero_point
  
X_valid = X_valid[:, :,np.newaxis]
X_valid = X_valid.astype('int8')



tflite_model_predictions=[]

for i in range(8):
  interpreter.set_tensor(input_details[0]['index'], X_valid[5000*i:5000*(i+1)])
  interpreter.invoke()
  Mod_Prediction=interpreter.get_tensor(output_details[0]['index'])
  if i==0:
    tflite_model_predictions=Mod_Prediction
  else:
    tflite_model_predictions=np.concatenate((tflite_model_predictions,Mod_Prediction),axis=0)
  print(i,"Done")

np.save('tflight_X_valid.npy',tflite_model_predictions)
#np.save('y_valid_s.npy',y_valid)
#print('Shape: ',tflite_model_predictions.shape)
#tflite_model_predictions=np.load('tflight_X_valid.npy')
#y_valid=np.load('y_valid_s.npy')
#tflite_model_predictions=np.load('tflite.npy')
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)

txt_1 = "Dist_{Class1:.0f}"
Distances={}
for i in range(NB_CLASSES):
  Distances[txt_1.format(Class1=i)]=[]
  
for i in range(len(tflite_model_predictions)):
  Distances[txt_1.format(Class1=y_valid[i])].append(open_prob[i][int(y_valid[i])])
   
TH=[0]*NB_CLASSES  
for j in range(NB_CLASSES):
  Distances[txt_1.format(Class1=j)].sort()
  Dist=Distances[txt_1.format(Class1=j)]
  if len(Dist)<15:
    print(len(Dist))
  TH[j]=Dist[int(len(Dist)*0.1)]  


Threasholds=np.array(TH)
np.save('Threasholds_openmax.npy',Threasholds)


print("Thresholds are calculated")

# //////////////////////////////////////////////////////////////////////////////

X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataNoDefCW()
del X_train,y_train,X_valid,y_valid

gc.collect()
# Please refer to the dataset format in readme
dataset_dir = "/home/ubuntu/Yasod/AWF/dataset/"

Data = np.load("/home/Yasod/Yasod/AWF/dataset/openset.npz",allow_pickle=True)
X_open=np.array(Data['data'])
X_open=X_open[:,:1500]
y_open = np.array([NB_CLASSES]*len(X_open))


X_test = X_test.astype('float32')
X_open = X_open.astype('float32')
y_test = y_test.astype('int16')

model=load_model('AWF_without_softmax.hdf5')
Mean_Vectors=np.load('Mean_vectors.npy')
Threasholds_openmax=np.load('Threasholds_openmax.npy')
Threasholds=np.load('Threasholds_Sdist.npy')

with open('/home/Yasod/Yasod/AWF/Openmax_trained', 'rb') as f:
    openmax_ob = pickle.load(f)

Mean_Vectors=np.load('Mean_vectors.npy')
Threasholds=np.load('Threasholds_Sdist.npy')
Threasholds_openmax=np.load('Threasholds_openmax.npy')

interpreter = tf.lite.Interpreter(model_path = 'tf_lite_fullint_softmax_openmax.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

Block_Size=5000

interpreter.resize_tensor_input(input_details[0]['index'], (Block_Size, 1500, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (Block_Size, NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_scale,input_zero_point  = input_details[0]["quantization"]
for i in range(len(X_test)):
  X_test[i] = X_test[i] / input_scale + input_zero_point
  
for i in range(len(X_open)):
  X_open[i] = X_open[i] / input_scale + input_zero_point


X_test = X_test[:, :,np.newaxis]
X_test = X_test.astype('int8')
X_open = X_open[:, :,np.newaxis]
X_open = X_open.astype('int8')


tflite_model_predictions = []
for i in range(8):
  interpreter.set_tensor(input_details[0]['index'], X_test[Block_Size*i:Block_Size*(i+1)])
  interpreter.invoke()
  Mod_Prediction=interpreter.get_tensor(output_details[0]['index'])
  if i==0:
    tflite_model_predictions=Mod_Prediction
  else:
    tflite_model_predictions=np.concatenate((tflite_model_predictions,Mod_Prediction),axis=0)
  print(i,"Done")

tflite_model_predictions=np.array(tflite_model_predictions)
np.save('tflight_close_set_predictions.npy',tflite_model_predictions)
np.save('y_test_quantized.npy',y_test)

#tflite_model_predictions=np.load('tflight_close_set_predictions.npy')
#y_test=np.load('y_test_quantized.npy')

prediction_classes=[]

open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)

prediction_classes=[]
for i in range(len(tflite_model_predictions)):
  
    d=np.argmax(tflite_model_predictions[i], axis=0)
    if np.linalg.norm(tflite_model_predictions[i]-Mean_Vectors[d])>Threasholds[d]:
      prediction_classes.append(NB_CLASSES)
      
    else:
      prediction_classes.append(d)
      
#***********************************************************
prediction_classes_openmax=[]

open_prob = openmax_ob.predict_prob_open(tflite_model_predictions)

for i in range(len(tflite_model_predictions)):
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes_openmax.append(d)
    elif open_prob[i][d]<Threasholds_openmax[d]:
      prediction_classes_openmax.append(NB_CLASSES)
      
    else:
      prediction_classes_openmax.append(d)
      
#************************************************************      
    

interpreter = tf.lite.Interpreter(model_path = 'tf_lite_fullint_softmax.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

interpreter.resize_tensor_input(input_details[0]['index'], (Block_Size, LENGTH, 1))
interpreter.resize_tensor_input(output_details[0]['index'], (Block_Size, NB_CLASSES))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



tflite_model_predictions_open = []
for i in range(38):
  interpreter.set_tensor(input_details[0]['index'], X_open[Block_Size*i:Block_Size*(i+1)])
  interpreter.invoke()
  Mod_Prediction=interpreter.get_tensor(output_details[0]['index'])
  if i==0:
    tflite_model_predictions_open=Mod_Prediction
  else:
    tflite_model_predictions_open=np.concatenate((tflite_model_predictions_open,Mod_Prediction),axis=0)
  print(i,"Done")

np.save('tflight_open_set_predictions.npy',tflite_model_predictions_open)
#tflite_model_predictions_open=np.load('tflight_open_set_predictions.npy')

prediction_classes_open=[]
for i in range(len(tflite_model_predictions_open)):
    d=np.argmax(tflite_model_predictions_open[i], axis=0)
    if np.linalg.norm(tflite_model_predictions_open[i]-Mean_Vectors[d])>Threasholds[d]:
      prediction_classes_open.append(NB_CLASSES)
      
    else:
      prediction_classes_open.append(d)

#*******************************************************************************
prediction_classes_open_openmax=[]
open_prob = openmax_ob.predict_prob_open(tflite_model_predictions_open)
for i in range(len(tflite_model_predictions_open)):
    d=np.argmax(open_prob[i], axis=0)
    if d==NB_CLASSES:
      prediction_classes_open_openmax.append(d)
    elif open_prob[i][d]<Threasholds_openmax[d]:
      prediction_classes_open_openmax.append(NB_CLASSES)
      
    else:
      prediction_classes_open_openmax.append(d)  
      
#***********************************************************************       
          
y_test=y_test[:len(prediction_classes)]
y_open=y_open[:len(prediction_classes_open)]
  
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
  
#print(Matrix)


print("Openmax_method results")





acc_Close = accuracy_score(prediction_classes_openmax, y_test[:len(prediction_classes_openmax)])
print('Test accuracy TFLITE model_Closed_set :', acc_Close)

acc_Open = accuracy_score(prediction_classes_open_openmax, y_open[:len(prediction_classes_open_openmax)])
print('Test accuracy TFLITE model_Open_set :', acc_Open)


Matrix=[]
for i in range(NB_CLASSES+1):
  Matrix.append(np.zeros(NB_CLASSES+1))
  
  
for i in range(len(y_test)):
  Matrix[y_test[i]][prediction_classes_openmax[i]]+=1
  
for i in range(len(y_open)):
  Matrix[y_open[i]][prediction_classes_open_openmax[i]]+=1
  
#print(Matrix)




F1_Score=New_F1_Score(Matrix)


#print("Average Precision: ", Precision)
#print("Average Recall: ", Recall)
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
print()


