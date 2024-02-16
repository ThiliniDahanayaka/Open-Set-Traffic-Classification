
import os
import pickle
import matplotlib.pyplot as plt
import itertools
import math


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras import backend as K
from utility import LoadDataIot
from Model_cnn import DCNet_Add_layer
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np
import tensorflow_addons as tfa
import tensorflow 
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay
from tensorflow.keras.models import Model
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics import accuracy_score
np_config.enable_numpy_behavior()

# import pickle

random.seed(0)

def representative_data_gen():
  for input_value in tensorflow.data.Dataset.from_tensor_slices(X_train_Rep).batch(1).take(100):
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


# Training the DF model
NB_EPOCH = 500   # Number of training epoch
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 64 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 500 # Packet sequence length
NB_CLASSES = 4 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)
Lambda=1
alpha=1
snapshot_number=8

#MyAdamW = extend_with_decoupled_weight_decay(Adam)
# Create a MyAdamW object
#OPTIMIZER = MyAdamW(weight_decay=0.001, learning_rate=0.0001,beta_1=0.9, beta_2=0.999)

# ...
learning_rate_data=[]
loss_data=[]


def step_decay(epoch):
   initial_lrate = 0.005
   
   if epoch<2 and i>0:
     lrate=0.01
   
   else:
     lrate = initial_lrate * (0.5+0.5*np.cos(epoch*math.pi/(NB_EPOCH//snapshot_number)))

   learning_rate_data.append(lrate)
   optimizer = model.optimizer
   optimizer.lr = lrate
   return lrate
   
lrate = LearningRateScheduler(step_decay)

OPTIMIZER = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None) # Optimizer




# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()


# Convert data as float32 type
X_train = X_train.astype('float32')
print(X_train[0][:300])
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')
# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]


y_train = to_categorical(y_train, NB_CLASSES)
y_valid = to_categorical(y_valid, NB_CLASSES)
#y_test = to_categorical(y_test, NB_CLASSES)

# Building and training model
print("Building and training DC model")

model = DCNet_Add_layer.build(input_shape=INPUT_SHAPE, nb_classes=NB_CLASSES)

filepath = 'DC_without_norm.hdf5'
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,lrate]
  
Predictions=[]

for i in range(snapshot_number):
  # callbacks_list.append(EarlyStopping(monitor='val_acc', mode='max', patience=6))
  #model.summary()
  
  history = model.fit(X_train, y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH//snapshot_number,verbose=VERBOSE, validation_data=(X_valid, y_valid), callbacks=callbacks_list)
  losss=history.history['val_loss']
  #print(losss.shape)
  loss_data+=losss
  
  Predictions.append(model.predict(X_test))
  
  model2 = Model(model.input, model.layers[-2].output)
  model2.save('snapshot_'+str(i)+'.hdf5')

Predictions=np.array(Predictions)
Avg_prediction=np.average(Predictions,axis=0)
print(Avg_prediction.shape)

with open('/home/Yasod/DC/His_Dict_without_Nom', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
 
Avg_prediction=np.argmax(Avg_prediction,axis=1)

score_test = accuracy_score(Avg_prediction, y_test)
print("Testing closed accuracy_without_norm:", score_test)

lr=history.history['lr']
learning_rate_data=np.array(learning_rate_data)
print(learning_rate_data.shape)
print(learning_rate_data[:10])

loss_data=np.array(loss_data)
print(loss_data.shape)

plt.figure()
plt.plot(learning_rate_data)
plt.show()

plt.figure()
plt.plot(loss_data)
plt.show()

#####################################################################################################################################

print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()



print("#####################################")


# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
#X_open = X_open.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('int8')

X_valid=np.concatenate((X_valid,X_train[-400:]),axis=0)
y_valid=np.concatenate((y_valid,y_train[-400:]),axis=0)

X_train=X_train[:-400]
y_train=y_train[:-400]

Mean_0=np.array([0]*NB_CLASSES)
Mean_1=np.array([0]*NB_CLASSES)
Mean_2=np.array([0]*NB_CLASSES)
Mean_3=np.array([0]*NB_CLASSES)
Mean_4=np.array([0]*NB_CLASSES)
Mean_5=np.array([0]*NB_CLASSES)

X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]


tflite_model_predictions=[]
for i in range(snapshot_number):
  model=load_model('snapshot_'+str(i)+'.hdf5')
  tflite_model_predictions.append(model.predict(X_train))
  
tflite_model_predictions=np.array(tflite_model_predictions)
tflite_model_predictions=np.average(tflite_model_predictions,axis=0)

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


tflite_model_predictions=[]
for i in range(snapshot_number):
  model=load_model('snapshot_'+str(i)+'.hdf5')
  tflite_model_predictions.append(model.predict(X_valid))
  
tflite_model_predictions=np.array(tflite_model_predictions)
tflite_model_predictions=np.average(tflite_model_predictions,axis=0)

Dist_0=[]
Dist_1=[]
Dist_2=[]
Dist_3=[]
Dist_4=[]
Dist_5=[]

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

Dist_0.sort()
Dist_1.sort()
Dist_2.sort()
Dist_3.sort()


percentile=0.90
Th_0=Dist_0[int(len(Dist_0)*percentile)]
Th_1=Dist_1[int(len(Dist_1)*percentile)]
Th_2=Dist_2[int(len(Dist_2)*percentile)]
Th_3=Dist_3[int(len(Dist_3)*percentile)]

Threasholds=np.array([Th_0,Th_1,Th_2,Th_3])
print(Threasholds)
np.save('./Temp/Threasholds.npy',Threasholds)


print("Thresholds are calculated")

# ///////////////////////////////////////////////////////////////////

print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()
# Please refer to the dataset format in readme
dataset_dir = "/home/Yasod/DC/dataset/"
X_open = np.load(dataset_dir+'X_open.npy')
y__open = np.load(dataset_dir+'y_open.npy')
print("y_open_shape: ",y__open.shape)
y_open=np.array([NB_CLASSES]*len(y__open))

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


Threasholds=np.load('./Temp/Threasholds.npy',allow_pickle=True)
Mean_Vectors=np.load('./Temp/Mean_vectors.npy')
print(Mean_Vectors.shape)
Mean_0=Mean_Vectors[0]
Mean_1=Mean_Vectors[1]
Mean_2=Mean_Vectors[2]
Mean_3=Mean_Vectors[3]

X_test = X_test[:, :,np.newaxis]
X_open = X_open[:, :,np.newaxis]

model=load_model('DC_without_softmax.hdf5')

tflite_model_predictions=[]
for i in range(snapshot_number):
  model=load_model('snapshot_'+str(i)+'.hdf5')
  tflite_model_predictions.append(model.predict(X_test))
  
tflite_model_predictions=np.array(tflite_model_predictions)
tflite_model_predictions=np.average(tflite_model_predictions,axis=0)

prediction_classes=[]
for i in range(len(tflite_model_predictions)):
  
    d=np.argmax(tflite_model_predictions[i], axis=0)
    if np.linalg.norm(tflite_model_predictions[i]-Mean_Vectors[d])>Threasholds[d]:
      prediction_classes.append(NB_CLASSES)
      
    else:
      prediction_classes.append(d)
      
prediction_classes=np.array(prediction_classes)  
acc_ = accuracy_score(prediction_classes, y_test)

# //////////////////////////////////////////////////////////////////////////////////////////////

tflite_model_predictions_open=[]
for i in range(snapshot_number):
  model=load_model('snapshot_'+str(i)+'.hdf5')
  tflite_model_predictions_open.append(model.predict(X_open))
  
tflite_model_predictions_open=np.array(tflite_model_predictions_open)
tflite_model_predictions_open=np.average(tflite_model_predictions_open,axis=0)

prediction_classes_open=[]
for i in range(len(tflite_model_predictions_open)):
  
    d=np.argmax(tflite_model_predictions_open[i], axis=0)
    if np.linalg.norm(tflite_model_predictions_open[i]-Mean_Vectors[d])>Threasholds[d]:
      prediction_classes_open.append(NB_CLASSES)
      
    else:
      prediction_classes_open.append(d)
      
  
prediction_classes_open=np.array(prediction_classes_open)  

print('Test accuracy Normal model_Closed_set :', acc_)
acc_ = accuracy_score(prediction_classes_open, y_open)
print('Test accuracy Normal model_Open_set :', acc_)



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



