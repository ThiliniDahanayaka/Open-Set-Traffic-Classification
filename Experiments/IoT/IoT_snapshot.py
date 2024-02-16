import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras import backend as K
from utility import LoadDataIot
from IOT_CNN import DFNet_Add_Layer
#from Model_DF import DFNet
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Model
import numpy as np
import os
from sklearn.metrics import accuracy_score


random.seed(0)


description = "Training and evaluating DF model for closed-world scenario on non-defended dataset"

print (description)
# Training the DF model
NB_EPOCH = 80   # Number of training epoch
print ("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 70 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 475 # Packet sequence length
OPTIMIZER = Adamax(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.003) # Optimizer

NB_CLASSES = 40 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)
snapshot_number=8


# ...
learning_rate_data=[]
loss_data=[]


def step_decay(epoch):
   initial_lrate = 0.08
   
   if epoch<2 and i>0:
     lrate=0.1
   
   else:
     lrate = initial_lrate * (0.5+0.5*np.cos(epoch*math.pi/(NB_EPOCH//snapshot_number)))

   learning_rate_data.append(lrate)
   optimizer = model.optimizer
   optimizer.lr = lrate
   return lrate
   
lrate = LearningRateScheduler(step_decay)

# Data: shuffled and split between train and test sets

print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataIot()
# Please refer to the dataset format in readme

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
#y_test = to_categorical(y_test, NB_CLASSES)
# Building and training model
# print ("Building and training DF model")

model = DFNet_Add_Layer.build(input_shape=INPUT_SHAPE, nb_classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])
print ("Model compiled")

filepath = 'IoT.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,lrate]

# Start training
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

print("############## Training is Done Successfully ###################")
model.save('IoT.hdf5')


Avg_prediction=np.argmax(Avg_prediction,axis=1)
print(Avg_prediction.shape)
        
#print("############## Training is Done Successfully ###################")
#model.save('DC_without_norm.hdf5')

#model=load_model('AWF.hdf5')

# Start evaluating model with testing data
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


np.save('./Temp/learning_rate_data.npy',learning_rate_data)
np.save('./Temp/loss_data.npy',loss_data)
np.save('./Temp/lr.npy',lr)

#####################################################################################################################################

