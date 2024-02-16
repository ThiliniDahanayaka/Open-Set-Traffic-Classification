
import numpy as np
from tensorflow.keras.utils import to_categorical
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def normalize(X,epsilon=1e-8):
		
    Raws=X.shape[0]
    Columns=X.shape[1]
    MAX=[]
    for i in range(Raws):
        MAX.append(max(X[i]))
                                        
    return X/(max(MAX)+epsilon)


dataset_dir = "./dataset/"
# # Load data for non-defended dataset for CW setting
def LoadDataIot():

    
    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    

    # Load training data
    X_train = np.load(dataset_dir+'X_train_5.npy',allow_pickle=True)
    y_train = np.load(dataset_dir+'y_train_5.npy',allow_pickle=True)
    # Load testing data
    X_test = np.load(dataset_dir+'X_test_5.npy',allow_pickle=True)
    y_test = np.load(dataset_dir+'y_test_5.npy',allow_pickle=True)

    # Load testing data
    X_valid = np.load(dataset_dir+'X_valid_5.npy',allow_pickle=True)
    y_valid = np.load(dataset_dir+'y_valid_5.npy',allow_pickle=True)
    
    
    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_test, y_test, X_valid, y_valid


def LoadDataIot_background():

    
    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    

    # Load training data
    X_train = np.load(dataset_dir+'X_train_5.npy',allow_pickle=True)
    y_train = np.load(dataset_dir+'y_train_5.npy',allow_pickle=True)
    # Load testing data
    X_test = np.load(dataset_dir+'X_test_5.npy',allow_pickle=True)
    y_test = np.load(dataset_dir+'y_test_5.npy',allow_pickle=True)

    # Load testing data
    X_valid = np.load(dataset_dir+'X_valid_5.npy',allow_pickle=True)
    y_valid = np.load(dataset_dir+'y_valid_5.npy',allow_pickle=True)
    
    X_open = np.load(dataset_dir+'X_open.npy',allow_pickle=True)
    y_open = np.load(dataset_dir+'y_open.npy',allow_pickle=True)
    X_open,y_open=shuffle(X_open, y_open)
    
    X_open_train, X_open_valid, y_open_train, y_open_valid = train_test_split(X_open, y_open, test_size=0.3, shuffle=False)
    X_open_valid, X_open_test, y_open_valid, y_open_test = train_test_split(X_open_valid, y_open_valid, test_size=0.3, shuffle=False)
    
    X_train=np.concatenate((X_train,X_open_train),axis=0)
    X_valid=np.concatenate((X_valid,X_open_valid),axis=0)
    #X_test=np.concatenate((X_test,X_open_test),axis=0)
    
    y_train=np.concatenate((y_train,y_open_train),axis=0)
    y_valid=np.concatenate((y_valid,y_open_valid),axis=0)
    #y_test=np.concatenate((y_test,y_open_test),axis=0)
    
    X_train,y_train=shuffle(X_train, y_train)
    X_valid,y_valid=shuffle(X_valid, y_valid)
    
    np.save(dataset_dir+'X_open_test.npy',X_open_test)
    np.save(dataset_dir+'y_open_test.npy',y_open_test)
    
    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_test, y_test, X_valid, y_valid
    
    

def Test_input_data():

    dict1={}
    txt_O = "X_Mean_{Class:.0f}"
    txt_1 = "Class_{Number:.0f}_Mean_Vector"
    for j in range(10):
      dict1[txt_O.format(Class=j)]=np.array([0]*500)


    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    


    # Load training data
    X_train = np.load(dataset_dir+'X_train.npy')
    y_train = np.load(dataset_dir+'y_train.npy')

    #X_train=normalize(X_train,epsilon=1e-8)
    
    
    for i in range(len(X_train)):
      dict1[txt_O.format(Class=y_train[i])]=np.sum([dict1[txt_O.format(Class=y_train[i])],X_train[i]], axis=0)
      
    for k in range(10):
      dict1[txt_O.format(Class=k)]=dict1[txt_O.format(Class=k)]/list(y_train).count(k)
      
    fig, axs = plt.subplots(2)
    fig.suptitle('Class 0 and Class 5')
    
    axs[0].plot(dict1[txt_O.format(Class=0)])
    axs[0].set_ylim([0, 350])
    axs[0].set(ylabel="Class_0")
    
    axs[1].plot(dict1[txt_O.format(Class=5)])
    axs[1].set_ylim([0, 350])
    axs[1].set(ylabel="Class_5")
        
    plt.show()
    
    
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Class 1 and Class 6')
    
    axs[0].plot(dict1[txt_O.format(Class=1)])
    axs[0].set_ylim([0, 350])
    axs[0].set(ylabel="Class_1")
    
    axs[1].plot(dict1[txt_O.format(Class=6)])
    axs[1].set_ylim([0, 350])
    axs[1].set(ylabel="Class_6")
        
    plt.show()
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Class 0 and Class 7')
    
    axs[0].plot(dict1[txt_O.format(Class=0)])
    axs[0].set_ylim([0, 350])
    axs[0].set(ylabel="Class_0")
    
    axs[1].plot(dict1[txt_O.format(Class=7)])
    axs[1].set_ylim([0, 350])
    axs[1].set(ylabel="Class_7")
        
    plt.show()
    
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Class 0 and Class 8')
    
    axs[0].plot(dict1[txt_O.format(Class=0)])
    axs[0].set_ylim([0, 350])
    axs[0].set(ylabel="Class_0")
    
    axs[1].plot(dict1[txt_O.format(Class=8)])
    axs[1].set_ylim([0, 350])
    axs[1].set(ylabel="Class_8")
        
    plt.show()
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Class 3 and Class 9')
    
    axs[0].plot(dict1[txt_O.format(Class=3)])
    axs[0].set_ylim([0, 350])
    axs[0].set(ylabel="Class_3")
    
    axs[1].plot(dict1[txt_O.format(Class=9)])
    axs[1].set_ylim([0, 350])
    axs[1].set(ylabel="Class_9")
        
    plt.show()

    
#Test_input_data()   
    
  
