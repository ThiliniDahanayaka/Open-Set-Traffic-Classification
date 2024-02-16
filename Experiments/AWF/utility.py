import pickle
import numpy as np
from numpy import load
import zipfile
from sklearn.utils import shuffle

NB_CLASSES = 200




# Load data for non-defended dataset for CW setting
def Reduce_Classes(X,y,N):
    length=len(y)
    New_X=[]
    New_Y=[]
    X=X.tolist()
    y=y.tolist()
    for i in range(length):
      if y[i]<N:
          New_X.append(X[i])
          New_Y.append(y[i])
        

    New_X=np.array(New_X)
    New_Y=np.array(New_Y)
    return New_X,New_Y
    

def LoadDataNoDefCW():
    
    
    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = "/content/dataset/dataset/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    
    X_train = load(dataset_dir+'X_train.npy')
    X_train = X_train[:, 0:1500]
    y_train = load(dataset_dir+'y_train.npy')
    
    # Load validation data
    X_valid = load(dataset_dir+'X_valid.npy')
    X_valid = X_valid[:, 0:1500]
    y_valid = load(dataset_dir+'y_valid.npy')
    X_valid,y_valid=shuffle(X_valid, y_valid)
    
    
    # Load testing data
    X_test = load(dataset_dir+'X_test.npy')
    X_test = X_test[:, 0:1500]
    y_test = load(dataset_dir+'y_test.npy')
    #X_test,y_test=shuffle(X_test, y_test)

    
    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)
    
    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    
def LoadDataNoDefCW_background():
    
    
    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = "/content/dataset/dataset/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    
    X_train = load(dataset_dir+'X_train.npy')
    X_train = X_train[:, 0:1500]
    y_train = load(dataset_dir+'y_train.npy')
    
    # Load validation data
    X_valid = load(dataset_dir+'X_valid.npy')
    X_valid = X_valid[:, 0:1500]
    y_valid = load(dataset_dir+'y_valid.npy')
    
    
    
    # Load testing data
    X_test = load(dataset_dir+'X_test.npy')
    X_test = X_test[:, 0:1500]
    y_test = load(dataset_dir+'y_test.npy')
    #X_test,y_test=shuffle(X_test, y_test)
    
    Data_openset = np.load("/home/Yasod/Yasod/AWF/dataset/openset.npz",allow_pickle=True)
    X_open=np.array(Data_openset['data'])
    X_open=X_open[:,:1500]
    
    X_train = np.concatenate(X_train,X_open[:len(X_train)],axis=0)
    X_open=X_open[len(X_train):]
    y_train = np.concatenate(y_train,[NB_CLASSES]*len(y_train),axis=0)
    X_train,y_train=shuffle(X_train, y_train)
    
    X_valid = np.concatenate(X_valid,X_open[:len(X_valid)],axis=0)
    X_open=X_open[len(X_valid):]
    y_valid = np.concatenate(y_valid,[NB_CLASSES]*len(y_valid),axis=0)
    X_valid,y_valid=shuffle(X_valid, y_valid)
    
    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)
    
    np.save('/home/Yasod/Yasod/AWF/dataset/X_open_rest.py',X_open)
    
    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def representative_data_gen(X_train, y_train):
    X_train_Rep,y_train_Rep=shuffle(X_train, y_train)
    for input_value in tf.data.Dataset.from_tensor_slices(X_train_Rep).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]
        
def quant_invoke(interpreter,input_details,output_details,X,Block,iter):
    tflite_model_predictions = []
    for i in range(iter):
      interpreter.set_tensor(input_details[0]['index'], X[Block*i:Block*(i+1)])
      interpreter.invoke()
      Mod_Prediction=interpreter.get_tensor(output_details[0]['index'])
      if i==0:
        tflite_model_predictions=Mod_Prediction
      else:
        tflite_model_predictions=np.concatenate((tflite_model_predictions,Mod_Prediction),axis=0)
      print(i,"Done. ",iter-i,"to go")
    return np.array(tflite_model_predictions)

def Mean_vector_calc(tflite_model_predictions,y_train):
    txt_O = "Mean_{Class1:.0f}"
    Means={}
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)
                
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
    return np.array(Mean_Vectors)
    

        
def Quantize(model,TF_LITE_MODEL_FILE_NAME,Block,LENGTH):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_model_name = TF_LITE_MODEL_FILE_NAME
    open(tflite_model_name, "wb").write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details() 

    interpreter.resize_tensor_input(input_details[0]['index'], (Block, LENGTH, 1))
    interpreter.resize_tensor_input(output_details[0]['index'], (Block, NB_CLASSES))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale,input_zero_point  = input_details[0]["quantization"]
    
    return interpreter,input_scale,input_zero_point



def train_AWF(model,OPTIMIZER,X_train, y_train,X_valid, y_valid,BATCH_SIZE,NB_EPOCH ):
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)


    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
        metrics=["accuracy"])
    print ("Model compiled")

    filepath = 'AWF_withoutNorm.hdf5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Start training
    print("############## Starting Training ###################")
    history = model.fit(X_train, y_train,
            batch_size=BATCH_SIZE, epochs=NB_EPOCH,
            verbose=2, validation_data=(X_valid, y_valid), callbacks=callbacks_list)

    with open('/disks/SATA_1/bi/Yasod/AWF/His_Dict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
    print("############## Training is Done Successfully ###################")
    model.save('AWF_normal.hdf5')
    return model
    
def train_AWF_quant(model,OPTIMIZER,X_train, y_train,X_valid, y_valid,X_test,y_test,BATCH_SIZE,NB_EPOCH, pre_trained_model ):
    y_train=np.argmax(y_train,axis=0)
    y_valid=np.argmax(y_valid,axis=0)
    y_test=np.argmax(y_test,axis=0)
    
    if pre_trained_model:
        model=load_model('AWF_normal.hdf5')
    else:
        interval_min = -128
        interval_max = 127
        y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train)) * (interval_max - interval_min) + interval_min
        y_valid = (y_valid - np.min(y_valid)) / (np.max(y_valid) - np.min(y_valid)) * (interval_max - interval_min) + interval_min
        model =train_AWF(model,OPTIMIZER,X_train, y_train,X_valid, y_valid,BATCH_SIZE,NB_EPOCH )
        
    TF_LITE_MODEL_FILE_NAME = "tf_lite_model.tflite"

    interpreter,input_scale,input_zero_point=Quantize(model,TF_LITE_MODEL_FILE_NAME,5000,1500)
    
    X_test = X_test[:, :,0]
    X_test = X_test / input_scale + input_zero_point
    X_test = X_test[:, :,np.newaxis]
    X_test = X_test.astype('int8')
    
    tflite_model_predictions=[]
    for i in range(8):
        interpreter.set_tensor(input_details[0]['index'], X_test[5000*i:5000*(i+1)])
        interpreter.invoke()
        Mod_Prediction=interpreter.get_tensor(output_details[0]['index'])
        prediction_classes = list(np.argmax(Mod_Prediction, axis=1))
        tflite_model_predictions = tflite_model_predictions+prediction_classes
        print(i,"Done. ",8-i,"to go")
        
    acc_ = accuracy_score(tflite_model_predictions, y_test)
    print('Test accuracy of the quantized model :', acc_)
    
    
def mean_threshold_gen(model_prediction_train,model_prediction_valid):
    y_train=np.argmax(y_train,axis=0)
    y_valid=np.argmax(y_valid,axis=0)
    model_predictions = model_prediction_train
    
    txt_O = "Mean_{Class1:.0f}"
    Means={}
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)
        
    count=[0]*NB_CLASSES

    for i in range(len(model_predictions)):
        #k=np.argmax(model_predictions[i])
        if (np.argmax(model_predictions[i])==y_train[i]):
            Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])]+model_predictions[i]
            count[y_train[i]]+=1
    print("Counts: ",count)

    Mean_Vectors=[]   
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
        Mean_Vectors.append(Means[txt_O.format(Class1=i)])

    Mean_vectors=np.array(Mean_Vectors)
    #np.save('Mean_vectors.npy', Mean_vectors, allow_pickle=True)
    
    model_predictions=model_prediction_valid

    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]
    
    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Distances[txt_1.format(Class1=y_valid[i])].append(dist)

    print(Distances)    
    TH=[0]*NB_CLASSES  
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        TH[j]=Dist[int(len(Dist)*0.9)]  



    Threasholds=np.array(TH)
    print(Threasholds)
    #np.save('Threasholds.npy',Threasholds)
    
    return Mean_vectors,Threasholds



def mean_threshold_gen_quant(model, X_train,y_train, X_valid, y_valid):
    y_train=np.argmax(y_train,axis=0)
    y_valid=np.argmax(y_valid,axis=0)
    X_train=X_train[:,:,0]
    X_valid=X_valid[:,:,0]
    y_train = y_train.astype('int16')
    y_valid = y_valid.astype('int16')
    y_test = y_test.astype('int16')


    TF_LITE_MODEL_FILE_NAME = "tf_lite_fullint_softmax.tflite"
    Block=5000
    LENGTH=1500
    interpreter,input_scale,input_zero_point = Quantize(model,TF_LITE_MODEL_FILE_NAME,Block,LENGTH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    X_train = X_train / input_scale + input_zero_point
    X_train = X_train[:, :,np.newaxis]
    X_train = X_train.astype('int8')

    tflite_model_predictions=quant_invoke(interpreter,input_details,output_details,X_train,Block,4)
    Mean_Vectors=Mean_vector_calc(tflite_model_predictions,y_train)

    X_valid[i] = X_valid[i] / input_scale + input_zero_point
    X_valid = X_valid[:, :,np.newaxis]
    X_valid = X_valid.astype('int8')
    tflite_model_predictions=quant_invoke(interpreter,input_details,output_details,X_valid,Block,8)
    
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
    print(Threasholds)
    np.save('Threasholds_Sdist.npy',Threasholds)
    
    return Mean_Vectors,Threasholds


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




def KPI_normal(model, X_test,y_test,X_open,y_open, Mean_Vectors,Threasholds):
    y_test=np.argmax(y_test,axis=0)
    y_open=np.argmax(y_open,axis=0)
    prediction_classes=[]
    tflite_model_predictions=model.predict(X_test)
    for i in range(len(tflite_model_predictions)):
    
        d=np.argmax(tflite_model_predictions[i], axis=0)
        if np.linalg.norm(tflite_model_predictions[i]-Mean_Vectors[d])>Threasholds[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)
        
    prediction_classes_open=[]
    tflite_model_predictions_open=model.predict(X_open)
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
    
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)
    

def KPI_quant(X_test,y_test,X_open,y_open, Mean_Vectors,Threasholds):
    X_test=X_test[:,:,0]
    X_open=X_open[:,:,0]
    y_test=np.argmax(y_test,axis=0)
    y_open=np.argmax(y_open,axis=0)
    interpreter = tf.lite.Interpreter(model_path = 'tf_lite_fullint_softmax.tflite')
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details() 

    Block_Size=5000

    interpreter.resize_tensor_input(input_details[0]['index'], (Block_Size, 1500, 1))
    interpreter.resize_tensor_input(output_details[0]['index'], (Block_Size, NB_CLASSES))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale,input_zero_point  = input_details[0]["quantization"]
    
    X_test[i] = X_test[i] / input_scale + input_zero_point
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
        print(i,"Done. ",7-i,"to go")

    tflite_model_predictions=np.array(tflite_model_predictions)
    np.save('tflight_closet_predictions.npy',tflite_model_predictions)

    prediction_classes=[]
    for i in range(len(tflite_model_predictions)):
    
        d=np.argmax(tflite_model_predictions[i], axis=0)
        if np.linalg.norm(tflite_model_predictions[i]-Mean_Vectors[d])>Threasholds[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)

    tflite_model_predictions_open = []
    for i in range(79):
      interpreter.set_tensor(input_details[0]['index'], X_open[Block_Size*i:Block_Size*(i+1)])
      interpreter.invoke()
      Mod_Prediction=interpreter.get_tensor(output_details[0]['index'])
      if i==0:
        tflite_model_predictions_open=Mod_Prediction
      else:
        tflite_model_predictions_open=np.concatenate((tflite_model_predictions_open,Mod_Prediction),axis=0)
      print(i,"Done. ",78-i,"to go")

    tflite_model_predictions_open = np.load("tflight_open_set_predictions.npy")

    prediction_classes_open=[]
    for i in range(len(tflite_model_predictions_open)):
        d=np.argmax(tflite_model_predictions_open[i], axis=0)
        if np.linalg.norm(tflite_model_predictions_open[i]-Mean_Vectors[d])>Threasholds[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
        
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
    
    for i in range(len(prediction_classes_open)):
        Matrix[y_open[i]][prediction_classes_open[i]]+=1
    
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)



################################    N dist methods    ###############################

def D1_calc(y_valid,model_predictions,Mean_Vectors):
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            #dist=np.dot(Mean_Vectors[y_valid[i]],tflite_model_predictions[i])
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Distances[txt_1.format(Class1=y_valid[i])].append(dist)

    #print(Distances)    
    TH=[0]*NB_CLASSES  
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        TH[j]=Dist[int(len(Dist)*0.92)]  

    Threasholds_D1=np.array(TH)
    return Threasholds_D1
    

def D2_calc(y_valid,model_predictions,Mean_Vectors,Indexes):
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            #dist=np.dot(Mean_Vectors[y_valid[i]],tflite_model_predictions[i])
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    #Tot+=np.dot(Mean_Vectors[k],tflite_model_predictions[i])-dist
                    Tot+=(np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist)
                Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    #print(Distances)    
    TH=[0]*NB_CLASSES  
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        TH[j]=Dist[int(len(Dist)*0.1)]  

    Threasholds_D2=np.array(TH)
    return Threasholds_D2

   
def D3_calc(y_valid,model_predictions,Mean_Vectors,Indexes):
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            #dist=np.dot(Mean_Vectors[y_valid[i]],tflite_model_predictions[i])  
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    #Tot=np.dot(Mean_Vectors[k],tflite_model_predictions[i])-dist
                    Tot+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])
                Tot=dist/Tot
                Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    #print(Distances)    
    TH=[0]*NB_CLASSES  
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        TH[j]=Dist[int(len(Dist)*0.92)]  

    Threasholds_D3=np.array(TH)
    return Threasholds_D3


def mean_threashold_gen_N(model, X_train,y_train, X_valid, y_valid):
    txt_O = "Mean_{Class1:.0f}"
    Means={}
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)
    
    model_predictions=model.predict(X_train)


    count=[0]*NB_CLASSES

    for i in range(len(model_predictions)):
        k=np.argmax(model_predictions[i])
        if (np.argmax(model_predictions[i])==y_train[i]):
            Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])]+model_predictions[i]
            count[y_train[i]]+=1

    Mean_Vectors=[]   
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
        Mean_Vectors.append(Means[txt_O.format(Class1=i)])

    Mean_vectors=np.array(Mean_Vectors)

    model_predictions=model.predict(X_valid)

    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]
    
    
    Indexes=[]
    for i in range(NB_CLASSES):
        Indexes.append([])
    Values={}
    for i in range(NB_CLASSES):
        Values[i]=[0]*NB_CLASSES
    
    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])

            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]):
                    Values[y_valid[i]][k]+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist 
        
    for i in range(NB_CLASSES):
        Tot=0
        for l in range(50):
            Min=min(Values[i])
            Tot+=Min
            Indexes[i].append(Values[i].index(Min))
            Values[i][Values[i].index(Min)]=1000000

    Indexes=np.array(Indexes)
    
    Threasholds_D1=D1_calc(y_valid,model_predictions,Mean_Vectors)
    Threasholds_D2=D2_calc(y_valid,model_predictions,Mean_Vectors,Indexes)
    Threasholds_D3=D3_calc(y_valid,model_predictions,Mean_Vectors,Indexes)
    
    return Mean_vectors, Threasholds_D1, Threasholds_D2, Threasholds_D3,Indexes


def mean_threashold_gen_N_quant(model, X_train,y_train, X_valid, y_valid):
    X_train=X_train[:,:,0]
    X_valid=X_valid[:,:,0]

    TF_LITE_MODEL_FILE_NAME = "tf_lite_fullint_softmax_openmax.tflite"

    Block,LENGTH=5000,1500
    interpreter,input_scale,input_zero_point=Quantize(model,TF_LITE_MODEL_FILE_NAME,Block,LENGTH)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details() 

    X_train = X_train / input_scale + input_zero_point
    
    X_train = X_train[:, :,np.newaxis]
    X_train = X_train.astype('int8')
    
    tflite_model_predictions=quant_invoke(interpreter,input_details,output_details,X_train,Block,6)

    Mean_Vectors=Mean_vector_calc(tflite_model_predictions,y_train)
    
    X_valid[i] = X_valid[i] / input_scale + input_zero_point
    X_valid = X_valid[:, :,np.newaxis]
    X_valid = X_valid.astype('int8')
    
    tflite_model_predictions=quant_invoke(interpreter,input_details,output_details,X_valid,Block,6)

    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]
    
    Indexes=[]
    for i in range(NB_CLASSES):
        Indexes.append([])
    
    Values={}
    for i in range(NB_CLASSES):
        Values[i]=[0]*NB_CLASSES
    
    for i in range(len(tflite_model_predictions)):
        if (y_valid[i]==np.argmax(tflite_model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-tflite_model_predictions[i])
            
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]):
                    Values[y_valid[i]][k]+=np.linalg.norm(Mean_Vectors[k]-tflite_model_predictions[i])-dist
        
    for i in range(NB_CLASSES):
        Tot=0
        for l in range(50):
            Min=min(Values[i])
            Tot+=Min
            Indexes[i].append(Values[i].index(Min))
            Values[i][Values[i].index(Min)]=1000000

    Indexes=np.array(Indexes)
    
    Threasholds_D1=D1_calc(y_valid,tflite_model_predictions,Mean_Vectors)
    Threasholds_D2=D2_calc(y_valid,tflite_model_predictions,Mean_Vectors,Indexes)
    Threasholds_D3=D3_calc(y_valid,tflite_model_predictions,Mean_Vectors,Indexes)
    
    return Mean_Vectors, Threasholds_D1, Threasholds_D2, Threasholds_D3,Indexes

def pred_classes_D1(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds):
    prediction_classes=[]
    for i in range(len(tflite_model_predictions)):
        d=np.argmax(tflite_model_predictions[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-tflite_model_predictions[i])  
  
        if dist>Threasholds[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)

    prediction_classes_open=[]
    for i in range(len(tflite_model_predictions_open)):
        d=np.argmax(tflite_model_predictions_open[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-tflite_model_predictions_open[i])
        
        if dist>Threasholds[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
            
    return prediction_classes,prediction_classes_open

def pred_classes_D2(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D2,Indexes):
    prediction_classes=[]
    for i in range(len(tflite_model_predictions)):
    
        d=np.argmax(tflite_model_predictions[i], axis=0)
        #dist=np.dot(Mean_vectors[d],tflite_model_predictions[i])
        dist=np.linalg.norm(Mean_vectors[d]-tflite_model_predictions[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d and (k in Indexes[d]):
            #Tot+=np.dot(Mean_vectors[k],tflite_model_predictions[i])-dist
                Tot+=np.linalg.norm(Mean_vectors[k]-tflite_model_predictions[i])-dist
            
        if Tot<Threasholds_D2[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)
        
    prediction_classes_open=[]
    for i in range(len(tflite_model_predictions_open)):
        d=np.argmax(tflite_model_predictions_open[i], axis=0)
        #dist=np.dot(Mean_vectors[d],tflite_model_predictions_open[i])
        dist = np.linalg.norm(Mean_vectors[d]-tflite_model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
            #Tot+=np.dot(Mean_vectors[k],tflite_model_predictions_open[i])-dist
                Tot+=np.linalg.norm(Mean_vectors[k]-tflite_model_predictions_open[i])-dist
            
        if Tot<Threasholds_D2[d]:
            prediction_classes_open.append(NB_CLASSES)
        
        else:
            prediction_classes_open.append(d)
            
    return prediction_classes,prediction_classes_open

def pred_classes_D3(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D3,Indexes):
    prediction_classes=[]
    for i in range(len(tflite_model_predictions)):
    
        d=np.argmax(tflite_model_predictions[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d][d]-tflite_model_predictions[i][d])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d and (k in Indexes[d]):
                Tot+=np.linalg.norm(Mean_vectors[k]-tflite_model_predictions[i])
            
        Tot=dist/Tot
        if Tot>Threasholds_D3[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)  

    prediction_classes_open=[]
    for i in range(len(tflite_model_predictions_open)):
        d=np.argmax(tflite_model_predictions_open[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-tflite_model_predictions_open[i])
        #dist=np.dot(Mean_vectors[d],tflite_model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
                Tot+=np.linalg.norm(Mean_vectors[k]-tflite_model_predictions_open[i])
                
        Tot=dist/Tot   
        if Tot>Threasholds_D3[d]:
            prediction_classes_open.append(NB_CLASSES)
        
        else:
            prediction_classes_open.append(d)
    return prediction_classes,prediction_classes_open


def build_matrix(y_test,y_open,prediction_classes,prediction_classes_open):
    Matrix=[]
    for i in range(NB_CLASSES+1):
        Matrix.append(np.zeros(NB_CLASSES+1))
    
    for i in range(len(y_test)):
        Matrix[y_test[i]][prediction_classes[i]]+=1
    
    for i in range(len(y_open)):
        Matrix[y_open[i]][prediction_classes_open[i]]+=1
    return Matrix


def KPI_normal_N(y_test,y_open,tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D1, Threasholds_D2, Threasholds_D3,Indexes):
    
    print()
    print("##################################### Distance Method 1 ###################################")
    print()
    prediction_classes,prediction_classes_open=pred_classes_D1(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D1)
    
    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy TFLITE model_Closed_set :', acc_Close)
    print('Test accuracy TFLITE model_Open_set :', acc_Open)
    
    Matrix=build_matrix(y_test,y_open,prediction_classes,prediction_classes_open)
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)

    print()
    print("##################################### Distance Method 2 ###################################")
    print()

    prediction_classes,prediction_classes_open=pred_classes_D2(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D2,Indexes) 
        
    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy TFLITE model_Closed_set :', acc_Close)
    print('Test accuracy TFLITE model_Open_set :', acc_Open)
    
    Matrix=build_matrix(y_test,y_open,prediction_classes,prediction_classes_open)
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)

    
    print()
    print("##################################### Distance Method 3 ###################################")
    print()

    prediction_classes,prediction_classes_open=pred_classes_D3(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D3,Indexes) 
        
    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy TFLITE model_Closed_set :', acc_Close)
    print('Test accuracy TFLITE model_Open_set :', acc_Open)
    
    Matrix=build_matrix(y_test,y_open,prediction_classes,prediction_classes_open)
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)


def KPI_quant_N(model,y_test,y_open,tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D1, Threasholds_D2, Threasholds_D3,Indexes):
    X_test=X_test[:,:,0]
    X_open=X_open[:,:,0]
    
    TF_LITE_MODEL_FILE_NAME = "tf_lite_fullint_softmax_openmax.tflite"
    Block,LENGTH=5000,1500

    interpreter,input_scale,input_zero_point  = Quantize(model,TF_LITE_MODEL_FILE_NAME,Block,LENGTH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details() 
    
    X_test[i] = X_test[i] / input_scale + input_zero_point
    X_open[i] = X_open[i] / input_scale + input_zero_point

    X_test = X_test[:, :,np.newaxis]
    X_test = X_test.astype('int8')
    X_open = X_open[:, :,np.newaxis]
    X_open = X_open.astype('int8')

    tflite_model_predictions = quant_invoke(interpreter,input_details,output_details,X_test,Block,6)
    tflite_model_predictions_open = quant_invoke(interpreter,input_details,output_details,X_open,Block,78)
    #np.save("tflight_open_set_predictions.npy",tflite_model_predictions_open)
    #tflite_model_predictions_open = np.load("tflight_open_set_predictions.npy")

    print()
    print("##################################### Distance Method 1 ###################################")
    print()
    prediction_classes,prediction_classes_open=pred_classes_D1(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D1)
    
    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy TFLITE model_Closed_set :', acc_Close)
    print('Test accuracy TFLITE model_Open_set :', acc_Open)
    
    Matrix=build_matrix(y_test,y_open,prediction_classes,prediction_classes_open)
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)

    print()
    print("##################################### Distance Method 2 ###################################")
    print()

    prediction_classes,prediction_classes_open=pred_classes_D2(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D2,Indexes) 
        
    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy TFLITE model_Closed_set :', acc_Close)
    print('Test accuracy TFLITE model_Open_set :', acc_Open)
    
    Matrix=build_matrix(y_test,y_open,prediction_classes,prediction_classes_open)
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)

    
    print()
    print("##################################### Distance Method 3 ###################################")
    print()

    prediction_classes,prediction_classes_open=pred_classes_D3(tflite_model_predictions,tflite_model_predictions_open,Mean_vectors,Threasholds_D3,Indexes) 
        
    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy TFLITE model_Closed_set :', acc_Close)
    print('Test accuracy TFLITE model_Open_set :', acc_Open)
    
    Matrix=build_matrix(y_test,y_open,prediction_classes,prediction_classes_open)
    F1_Score=Micro_F1(Matrix)
    print("Average F1_Score: ", F1_Score)
    


    
    

