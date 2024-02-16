
import json
import numpy as np
from tensorflow.keras.utils import to_categorical
import json
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def Exchange_classes(c1,c2,y):
  for i in range(len(y)):
    if y[i]==c1:
      y[i]=c2
      
    elif y[i]==c2:
      y[i]=c1

def background_change(X_train_5,X_valid_5,y_train_5, y_valid_5,X_open,y_open,data,Split_Number ):
    closed_set=data[Split_Number]["closed"]
    open_set=data[Split_Number]["open"]

    #print(len(X_train_5))
    X_open_train=[]
    y_open_train=[]
    X_open_new=[]
    y_open_new=[]
    for d in range(len(X_open)):
      if y_open[d]<55:
        X_open_train.append(X_open[d])
        y_open_train.append(40)
      
      else:
        X_open_new.append(X_open[d])
        y_open_new.append(40)
    
    X_train_5=np.concatenate((X_train_5,X_open_train[:int(len(X_open_train)*2/3)]),axis=0)
    X_valid_5=np.concatenate((X_valid_5,X_open_train[int(len(X_open_train)*2/3):]),axis=0)
    #X_test=np.concatenate((X_test,X_open_test),axis=0)
    #print(len(X_train_5))
    
    y_train_5=np.concatenate((y_train_5,y_open_train[:int(len(y_open_train)*2/3)]),axis=0)
    y_valid_5=np.concatenate((y_valid_5,y_open_train[int(len(y_open_train)*2/3):]),axis=0)
    
    X_train_5,y_train_5=shuffle(X_train_5, y_train_5)
    X_valid_5,y_valid_5=shuffle(X_valid_5, y_valid_5)
    
    
    np.save(dataset_dir+'X_open_test.npy',X_open_new)
    np.save(dataset_dir+'y_open_test.npy',y_open_new)
    
    return X_train_5,X_valid_5,y_train_5,y_valid_5
    

def balance_dataset(X,y):
  Classes=int(np.max(y))+1
  k=len(y)//Classes-1
  X_=list(np.zeros(k*Classes))
  y_=list(np.zeros(k*Classes))
  
  
  for j in range(Classes):
    for i in range(k):
      y_[i*Classes+j]=j
      Index=list(y).index(j)
      X_[i*Classes+j]=X[Index]
      y[Index]=1000
  
  y=np.array(y_)
  X=np.array(X_)
  return X,y
  
def Filter_close_set(Split_Number,data):
  closed_set=data[Split_Number]["closed"]
  open_set=data[Split_Number]["open"]
  X_5=[]
  X_open=[]
  y_5=[]
  y_open=[]
  for i in range(len(y)):
    if y[i] in closed_set:
      y_5.append(closed_set.index(y[i]))
      X_5.append(X[i])
    else:
      y_open.append(40+int(open_set.index(y[i])))
      X_open.append(X[i])
      
  return X_5,X_open,y_5,y_open
  
  
  
  
  
f = open('./dataset/IOT_data_file.json')
data = json.load(f)


dataset_dir = "/content/gdrive/My Drive/SydneyDatasets/IoT/"
X_train = np.load(dataset_dir+'X_train.npy',allow_pickle=True)

y_train = np.load(dataset_dir+'y_train.npy',allow_pickle=True)
#print(list(y_train).count(0),list(y_train).count(1),list(y_train).count(2),list(y_train).count(3),list(y_train).count(4),list(y_train).count(5),list(y_train).count(6),list(y_train).count(7),list(y_train).count(8),list(y_train).count(9))

    # Load testing data
X_test = np.load(dataset_dir+'X_test.npy',allow_pickle=True)
y_test = np.load(dataset_dir+'y_test.npy',allow_pickle=True)


    # Load testing data
X_valid = np.load(dataset_dir+'X_valid.npy',allow_pickle=True)
y_valid = np.load(dataset_dir+'y_valid.npy',allow_pickle=True)

X=np.concatenate((X_train, X_test,X_valid), axis=0)
y=np.concatenate((y_train, y_test,y_valid), axis=0)

  
#X,y=balance_dataset(X,y)


Split_Number="10"
  
  
X_5,X_open,y_5,y_open=Filter_close_set(Split_Number,data)


  
X_train_5, X_valid_5, y_train_5, y_valid_5 = train_test_split(X_5, y_5, test_size=0.3, shuffle=False)
X_valid_5, X_test_5, y_valid_5, y_test_5 = train_test_split(X_valid_5, y_valid_5, test_size=0.4, shuffle=False)

############  Background  ################  
#X_train_5,X_valid_5,y_train_5,y_valid_5=background_change(X_train_5,X_valid_5,y_train_5, y_valid_5,X_open,y_open,data,Split_Number)     

#############################

X_train_5,y_train_5=shuffle(X_train_5, y_train_5)
X_valid_5,y_valid_5=shuffle(X_valid_5, y_valid_5)
X_test_5,y_test_5=shuffle(X_test_5, y_test_5)


X_train_5=np.array(X_train_5)
X_valid_5=np.array(X_valid_5)
X_test_5=np.array(X_test_5)
y_train_5=np.array(y_train_5)
y_valid_5=np.array(y_valid_5)
y_test_5=np.array(y_test_5)
X_open=np.array(X_open)
y_open=np.array(y_open)


print ("Data dimensions:")
print ("X: Training data's shape : ", X_train_5.shape)
print ("X: Validating data's shape : ", X_valid_5.shape)
print ("X: Testing data's shape : ", X_test_5.shape)
print ("X: open data's shape : ", X_open.shape)
    
np.save(dataset_dir+'X_train_5.npy',X_train_5)
np.save(dataset_dir+'X_valid_5.npy',X_valid_5)
np.save(dataset_dir+'X_test_5.npy',X_test_5)
np.save(dataset_dir+'y_valid_5.npy',y_valid_5)
np.save(dataset_dir+'y_train_5.npy',y_train_5)
np.save(dataset_dir+'y_test_5.npy',y_test_5)
np.save(dataset_dir+'X_open.npy',X_open)
np.save(dataset_dir+'y_open.npy',y_open)

