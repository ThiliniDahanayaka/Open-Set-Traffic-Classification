import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras import backend as K
from utility import LoadDataIot
from Model_NoDef import DFNet
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import preprocessing as sp
import pandas as pd 
import numpy as np
import os



random.seed(0)


description = "Training and evaluating DF model for closed-world scenario on non-defended dataset"

print (description)
# Training the DF model
NB_EPOCH = 30   # Number of training epoch
print ("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 128 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 5000 # Packet sequence length
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

NB_CLASSES = 95 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)

snapshot_number=6

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
    
  for k in range(len(Recalls)):
    Recalls[k]=Matrix[k][k]/np.sum(Matrix,axis=1)[k]
   
  Recall=np.average(Recalls)

  F1_Score=2*Precision*Recall/(Precision+Recall+epsilon)
  return F1_Score

# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataIot()


# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')
# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to categorical classes matrices
y_train = to_categorical(y_train, NB_CLASSES)
y_valid = to_categorical(y_valid, NB_CLASSES)


model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])
print ("Model compiled")

filepath = 'DF.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,lrate]

# Start training
Predictions=[]

for i in range(snapshot_number):
  
  history = model.fit(X_train, y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH//snapshot_number,verbose=VERBOSE, validation_data=(X_valid, y_valid), callbacks=callbacks_list)
  losss=history.history['val_loss']
  
  loss_data+=losss
  
  Predictions.append(model.predict(X_test))
  
  model2 = Model(model.input, model.layers[-2].output)
  model2.save('snapshot_'+str(i)+'.hdf5')

print("############## Training is Done Successfully ###################")

Predictions=np.array(Predictions)
Avg_prediction=np.average(Predictions,axis=0)
Avg_prediction=np.argmax(Avg_prediction,axis=1)
        
# Start evaluating model with testing data
score_test = accuracy_score(Avg_prediction, y_test)
print("Testing closed accuracy_without_norm:", score_test)

learning_rate_data=np.array(learning_rate_data)
loss_data=np.array(loss_data)


plt.figure()
plt.plot(learning_rate_data)
plt.show()

plt.figure()
plt.plot(loss_data)
plt.show()

#####################################################################################################################################


X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()

X_valid,y_valid=shuffle(X_valid, y_valid)

# Please refer to the dataset format in readme
dataset_dir = "/home/ubuntu/Yasod/AWF/DF/dataset/"

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

#y_test_=np.argmax(y_test_, axis=1)
txt_O = "Mean_{Class1:.0f}"
Means={}
for i in range(NB_CLASSES):
  Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)
  
model=load_model('DF_without_softmax.hdf5',compile=True)

X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]

tflite_model_predictions = []
for i in range(snapshot_number):
  model=load_model('snapshot_'+str(i)+'.hdf5')
  tflite_model_predictions.append(model.predict(X_train))

tflite_model_predictions=np.array(tflite_model_predictions)
tflite_model_predictions=np.average(tflite_model_predictions,axis=0)

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

Mean_Vectors=np.array(Mean_Vectors)
np.save('./Temp/Mean_vectors.npy', Mean_Vectors, allow_pickle=True)



tflite_model_predictions=[]
for i in range(snapshot_number):
  model=load_model('snapshot_'+str(i)+'.hdf5')
  tflite_model_predictions.append(model.predict(X_valid))
  
tflite_model_predictions=np.array(tflite_model_predictions)
tflite_model_predictions=np.average(tflite_model_predictions,axis=0)


print('Shape: ',tflite_model_predictions.shape)

txt_1 = "Dist_{Class1:.0f}"
Distances={}
for i in range(NB_CLASSES):
  Distances[txt_1.format(Class1=i)]=[]
  
  
for i in range(len(tflite_model_predictions)):
  if (y_valid[i]==np.argmax(tflite_model_predictions[i])):
    dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-tflite_model_predictions[i])
    Distances[txt_1.format(Class1=y_valid[i])].append(dist)
    
TH=[0]*NB_CLASSES  
for j in range(NB_CLASSES):
  Distances[txt_1.format(Class1=j)].sort()
  Dist=Distances[txt_1.format(Class1=j)]
  TH[j]=Dist[int(len(Dist)*0.92)]  

Threasholds=np.array(TH)
np.save('./Temp/Threasholds.npy',Threasholds)

print("Thresholds are calculated")

# /////////////////////////////////////////////////////////////////////////

X_train, y_train, X_test, y_test, X_valid, y_valid  = LoadDataIot()
del X_train,y_train,X_valid,y_valid

gc.collect()
# Please refer to the dataset format in readme
dataset_dir = "/home/Yasod/Yasod/DF/dataset/"

with open( "/home/Yasod/Yasod/DF/dataset/X_open.pkl","rb") as f:
    X_open = cPickle.load(f,encoding='latin1')
    X_open = np.array(X_open)
    X_open = X_open[:, 0:5000]
y_open = np.array([NB_CLASSES]*len(X_open))

X_test,y_test=shuffle(X_test, y_test)


X_test = X_test.astype('float32')
X_open = X_open.astype('float32')
y_test = y_test.astype('int16')


Mean_Vectors=np.load('./Temp/./Temp/Mean_vectors.npy')
Threasholds=np.load('./Temp/./Temp/Threasholds.npy')

X_test = X_test[:, :,np.newaxis]
X_open = X_open[:, :,np.newaxis]


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




