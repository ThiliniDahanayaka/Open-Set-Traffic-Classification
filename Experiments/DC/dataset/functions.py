
def cal_mean_vector(X_train,y_train,model,NB_CLASSES):    #X_train dim:(nb_raws,length,1)
                                                          #y_train dim:(nb_raws,1)
    txt_O = "Mean_{Class1:.0f}"
    Means={}
    for i in range(NB_CLASSES):
      Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)


    model_predictions = model.predict(X_train)
    print('Shape: ',model_predictions.shape)
    count=[0]*NB_CLASSES

    for i in range(len(model_predictions)):
      k=np.argmax(model_predictions[i])
      if (np.argmax(model_predictions[i])==y_train[i]):
        Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])]+tflite_model_predictions[i]
        count[y_train[i]]+=1


    Mean_Vectors=[]   
    for i in range(NB_CLASSES):
      Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
      Mean_Vectors.append(Means[txt_O.format(Class1=i)])
    Mean_vectors=np.array(Mean_Vectors)
    return Mean_vectors


def cal_Thresholds(X_valid,y_valid,model,NB_CLASSES,Mean_Vectors):
  model_predictions=model.predict(X_valid)
  print('Shape: ',model_predictions.shape)

  txt_1 = "Dist_{Class1:.0f}"
  Distances={}
  for i in range(NB_CLASSES):
    Distances[txt_1.format(Class1=i)]=[]
    
    
  for i in range(len(model_predictions)):
    if (y_valid[i]==np.argmax(model_predictions[i])):
      dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
      Distances[txt_1.format(Class1=y_valid[i])].append(dist)
    
  TH=[0]*NB_CLASSES  
  for j in range(NB_CLASSES):
    Distances[txt_1.format(Class1=j)].sort()
    Dist=Distances[txt_1.format(Class1=j)]
    TH[j]=Dist[int(len(Dist)*0.90)]  

  Threasholds=np.array(TH)
  return Threasholds



def cal_Acc_matrix(X_test,y_test,model,Threasholds,Mean_Vectors):
  prediction_classes=[]
  model_predictions=model.predict(X_test)
  for i in range(len(model_predictions)):
      d=np.argmax(model_predictions[i], axis=0)
      if np.linalg.norm(model_predictions[i]-Mean_Vectors[d])>Threasholds[d]:
        prediction_classes.append(NB_CLASSES)  
      else:
        prediction_classes.append(d)    

  prediction_classes_open=[]
  model_predictions_open=model.predict(X_open)
  for i in range(len(model_predictions_open)):
      d=np.argmax(model_predictions_open[i], axis=0)
      if np.linalg.norm(model_predictions_open[i]-Mean_Vectors[d])>Threasholds[d]:
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

  return Matrix



def Micro_F1(matrix,NB_CLASSES):
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


  
