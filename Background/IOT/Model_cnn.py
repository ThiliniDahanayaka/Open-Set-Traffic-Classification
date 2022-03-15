from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Sequential

class DCNet:
    @staticmethod
    def build(input_shape, nb_classes=98, trainable=True):

        model = Sequential()

        # Feature Extractor
        model.add(Conv1D(128, kernel_size=7, activation='tanh', input_shape=input_shape, use_bias=False, kernel_initializer='glorot_normal', trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.1, trainable=trainable))

        model.add(Conv1D(128, kernel_size=19, activation='elu', kernel_initializer='glorot_normal', trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.3, trainable=trainable))

        model.add(Conv1D(64, kernel_size=13, activation='elu', kernel_initializer='glorot_normal', trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(Dropout(rate=0.1, trainable=trainable))

        model.add(Conv1D(256, kernel_size=23, activation='selu', kernel_initializer='glorot_normal', trainable=trainable))
        model.add(BatchNormalization(trainable=trainable))
        model.add(MaxPooling1D(1, trainable=trainable))
        model.add(GlobalAveragePooling1D(trainable=trainable))

        model.add(Dense(180, activation='selu', kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(150, activation='selu', kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(nb_classes, activation='softmax'))

        return model
