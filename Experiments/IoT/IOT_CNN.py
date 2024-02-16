from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D,Activation
from tensorflow.keras.models import Sequential

class DFNet:
    @staticmethod
    def build(input_shape, nb_classes=98, trainable=True):

        model = Sequential()

        # Feature Extractor
        model.add(Conv1D(128, kernel_size=7, activation='tanh', input_shape=input_shape, use_bias=False, trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.1, trainable=trainable))

        model.add(Conv1D(128, kernel_size=19, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.3, trainable=trainable))

        model.add(Conv1D(64, kernel_size=13, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.1, trainable=trainable))

        model.add(Conv1D(256, kernel_size=23, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(GlobalAveragePooling1D(trainable=trainable))

        model.add(Dense(180, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dense(150, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dense(nb_classes,name='phenultimate_layer'))
        model.add(Activation('softmax', name='softmax', trainable=trainable))

        return model
        
class DFNet_Dropout:
    @staticmethod
    def build(input_shape, nb_classes=98, trainable=True):

        model = Sequential()

        # Feature Extractor
        model.add(Conv1D(128, kernel_size=7, activation='tanh', input_shape=input_shape, use_bias=False, trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.1, trainable=trainable))

        model.add(Conv1D(128, kernel_size=19, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.3, trainable=trainable))

        model.add(Conv1D(64, kernel_size=13, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.3, trainable=trainable))

        model.add(Conv1D(256, kernel_size=23, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(GlobalAveragePooling1D(trainable=trainable))

        model.add(Dense(180, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.4, trainable=trainable))
        model.add(Dense(150, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dense(nb_classes,name='phenultimate_layer'))
        model.add(Activation('softmax', name='softmax', trainable=trainable))

        return model
        

class DFNet_Add_Layer:
    @staticmethod
    def build(input_shape, nb_classes=98, trainable=True):

        model = Sequential()

        # Feature Extractor
        model.add(Conv1D(128, kernel_size=7, activation='tanh', input_shape=input_shape, use_bias=False, trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.3, trainable=trainable))

        model.add(Conv1D(128, kernel_size=19, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.5, trainable=trainable))

        model.add(Conv1D(64, kernel_size=13, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.3, trainable=trainable))
        
        
        model.add(Conv1D(128, kernel_size=17, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.5, trainable=trainable))


        model.add(Conv1D(256, kernel_size=23, activation='relu',  trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(GlobalAveragePooling1D(trainable=trainable))

        model.add(Dense(180, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5, trainable=trainable))
        model.add(Dense(150, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dense(nb_classes,name='phenultimate_layer'))
        model.add(Activation('softmax', name='softmax', trainable=trainable))

        return model
