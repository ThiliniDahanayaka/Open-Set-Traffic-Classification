from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Flatten, Dense, Dropout

'''
from keras.optimizers import Adam

batch_size = 64
'''
class DCNet:
    @staticmethod
    def build(input_shape=[500, 1], nb_classes=10, trainable=True):

        filter_num = [None, 32, 32, 32]
        kernel_size = [None, 16, 16, 16]
        conv_stride_size = ['None', 1, 1, 1]
        pool_stride_size = ['None', 4]
        pool_size = ['None', 8]

        model = Sequential()

        # Feature Extractor
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                         strides=conv_stride_size[1], padding='valid', name='conv1', activation='relu', trainable=trainable))
        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2], strides=conv_stride_size[2], padding='valid',
                         name='conv2', activation='relu',trainable=trainable))
        model.add(Dropout(rate=0.3, name='cov1_dropout', trainable=trainable))

        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3], strides=conv_stride_size[3], padding='valid',
                         name='conv3', activation='relu', trainable=trainable))
        model.add(Dropout(rate=0.3, name='conv3_dropout', trainable=trainable))

        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1], padding='valid', name='max_pool',
                               trainable=trainable))
        
        model.add(Activation('relu', name='max_pool_act', trainable=trainable))
        model.add(Dropout(rate=0.3, name='max_pool_dropout', trainable=trainable))

        # Classifier
        model.add(Flatten(name='flatten'))
        model.add(Dense(64, activation='relu', name='fc1'))
        model.add(Dropout(rate=0.5, name='fc1_dropout'))
        model.add(Dense(nb_classes, name='fc_before_softmax'))
        model.add(Activation('softmax', name='softmax', trainable=trainable))

        return model