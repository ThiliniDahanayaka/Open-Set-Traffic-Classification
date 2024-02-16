from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
import tensorflow as tf

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
        pool_stride_size = ['None', 6]
        pool_size = ['None', 6]

        model = Sequential()

        # Feature Extractor
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,strides=conv_stride_size[1], padding='valid', name='conv1', activation='relu', trainable=trainable))
        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2], strides=conv_stride_size[2], padding='valid',name='conv2', activation='relu',trainable=trainable))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3], strides=conv_stride_size[3], padding='valid',name='conv3', activation='relu', trainable=trainable))
        model.add(Dropout(rate=0.5, name='cov_dropout', trainable=trainable))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1], padding='valid', name='max_pool',trainable=trainable))
        model.add(Activation('relu', name='max_pool_act', trainable=trainable))
        model.add(Dropout(rate=0.3, name='max_pool_dropout', trainable=trainable))

        # Classifier
        model.add(Flatten(name='flatten'))
        model.add(Dense(64, activation='relu', name='fc1'))
        model.add(Dropout(rate=0.5, name='fc1_dropout'))
        model.add(Dense(nb_classes, name='fc_before_softmax'))
        model.add(Activation('softmax', name='softmax', trainable=trainable))

        return model
        
        
class DCNet_Dropouts:
    @staticmethod
    def build(input_shape=[500, 1], nb_classes=10, trainable=True):
        filter_num = [None, 32, 32, 32]
        kernel_size = [None, 16, 16, 16]
        conv_stride_size = ['None', 1, 1, 1]
        pool_stride_size = ['None', 6]
        pool_size = ['None', 16]
        model=tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,strides=conv_stride_size[1], padding='valid', name='conv1', activation='relu', trainable=trainable),
            tf.keras.layers.Conv1D(filters=filter_num[2], kernel_size=kernel_size[2], strides=conv_stride_size[2], padding='valid',name='conv2', activation='relu',trainable=trainable),
            tf.keras.layers.Conv1D(filters=filter_num[3], kernel_size=kernel_size[3], strides=conv_stride_size[3], padding='valid',name='conv3', activation='relu', trainable=trainable),
            tf.keras.layers.Dropout(rate=0.6, name='cov_dropout', trainable=trainable),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1], padding='valid', name='max_pool',trainable=trainable),
            tf.keras.layers.Activation('relu', name='max_pool_act', trainable=trainable),
            tf.keras.layers.Dropout(rate=0.5, name='max_pool_dropout', trainable=trainable),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(64, activation='relu', name='fc1'),
            tf.keras.layers.Dropout(rate=0.6, name='fc1_dropout'),
            tf.keras.layers.Dense(nb_classes, name='fc_before_softmax'),
            tf.keras.layers.Activation('softmax', name='softmax', trainable=trainable),
            ])
        #model = tf.keras.Sequential()
        # Feature Extractor
        #model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
        #                 strides=conv_stride_size[1], padding='valid', name='conv1', activation='relu', trainable=trainable))
        #model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2], strides=conv_stride_size[2], padding='valid',
        #                 name='conv2', activation='relu',trainable=trainable))
        #model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3], strides=conv_stride_size[3], padding='valid',
        #                 name='conv3', activation='relu', trainable=trainable))
        #model.add(Dropout(rate=0.5, name='cov_dropout', trainable=trainable))
        #model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1], padding='valid', name='max_pool',
        #                       trainable=trainable))
        #model.add(Activation('relu', name='max_pool_act', trainable=trainable))
        #model.add(Dropout(rate=0.3, name='max_pool_dropout', trainable=trainable))
        # Classifier
        #model.add(Flatten(name='flatten'))
        #model.add(Dense(64, activation='relu', name='fc1'))
        #model.add(Dropout(rate=0.5, name='fc1_dropout'))
        #model.add(Dense(nb_classes, name='fc_before_softmax'))
        #model.add(Activation('softmax', name='softmax', trainable=trainable))
        return model

class DCNet_Add_layer:
    @staticmethod
    def build(input_shape=[500, 1], nb_classes=10, trainable=True):

        filter_num = [None, 32, 32, 32,64]
        kernel_size = [None, 16, 16, 16,64]
        conv_stride_size = ['None', 1, 1, 1]
        pool_stride_size = ['None', 6]
        pool_size = ['None', 6]

        model = Sequential()

        # Feature Extractor
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                         strides=conv_stride_size[1], padding='valid', name='conv1', activation='relu', trainable=trainable))
        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2], strides=conv_stride_size[2], padding='valid',
                         name='conv2', activation='relu',trainable=trainable))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3], strides=conv_stride_size[3], padding='valid',
                         name='conv3', activation='relu', trainable=trainable))
        model.add(Dropout(rate=0.6, name='cov_dropout_added', trainable=trainable))            
        
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4], strides=conv_stride_size[3], padding='valid',
                         name='conv4', activation='relu', trainable=trainable))               
        model.add(Dropout(rate=0.6, name='cov_dropout', trainable=trainable))
        
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1], padding='valid', name='max_pool',
                               trainable=trainable))
        model.add(Activation('relu', name='max_pool_act', trainable=trainable))
        model.add(Dropout(rate=0.5, name='max_pool_dropout', trainable=trainable))

        # Classifier
        model.add(Flatten(name='flatten'))
        model.add(Dense(64, activation='relu', name='fc1'))
        model.add(Dropout(rate=0.6, name='fc1_dropout'))
        model.add(Dense(nb_classes, name='fc_before_softmax'))
        model.add(Activation('softmax', name='softmax', trainable=trainable))

        return model


