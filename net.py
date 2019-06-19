# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.models import Model, clone_model
from keras.layers import Input, Dense, merge, Dropout, Activation, Flatten, Reshape, MaxPooling2D
from keras.layers.local import LocallyConnected2D
#from keras.layers.convolutional import Conv1D
#from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

def build_pretrain(input_dim, output_dim, hidden):

    net_pretrain = Sequential()
    net_pretrain.add(Dense(hidden, input_shape=(input_dim,)))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Dense(hidden))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Dense(hidden))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Dense(output_dim))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('softmax'))
    return net_pretrain

def build_pretrain_local(input_dim, output_dim):
    net_pretrain = Sequential()
    net_pretrain.add(Reshape((input_dim,1,1), input_shape=(input_dim,)))
    net_pretrain.add(LocallyConnected2D(20,
                  (5,1),
                  strides=(3,1),
                  padding='valid',
                  input_shape=(input_dim,1,1),
                  data_format='channels_last'))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(LocallyConnected2D(15,
                  (5,1),
                  strides=(3,1),
                  padding='valid',
                  data_format='channels_last'))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Flatten())
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Dense(100))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Dense(output_dim))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('softmax'))
    return net_pretrain

def build_pretrain_conv(input_dim, output_dim):
    net_pretrain = Sequential()
    net_pretrain.add(Reshape((input_dim,1,1), input_shape=(input_dim,)))
    net_pretrain.add(Conv2D(20,
                  (5,1),
                  strides=(3,1),
                  padding='valid',
                  input_shape=(input_dim,1,1),
                  data_format='channels_last'))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Conv2D(15,
                  (5,1),
                  strides=(3,1),
                  padding='valid',
                  data_format='channels_last'))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Flatten())
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Dense(100))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('selu'))
    net_pretrain.add(Dropout(0.5))
    net_pretrain.add(Dense(output_dim))
    net_pretrain.add(BatchNormalization())
    net_pretrain.add(Activation('softmax'))
    return net_pretrain

# [Deep Convolutional Neural Networks for Hyperspectral Image Classification] by Wei Hu etc.
def build_pretrain_cnn(input_dim, output_dim, network_type):
    net_pretrain = Sequential()
    net_pretrain.add(Reshape((input_dim,1,1), input_shape=(input_dim,)))
    if network_type == 0:
        net_pretrain.add(Conv2D(20,
              (24,1),
              # strides=(13,1),
              padding='valid',
              input_shape=(input_dim,1,1),
              data_format='channels_last'))
        net_pretrain.add(Activation('tanh'))
        net_pretrain.add(MaxPooling2D((5,1), data_format='channels_last'))
        net_pretrain.add(Flatten())
        net_pretrain.add(Dense(100))
        net_pretrain.add(Activation('tanh'))
        net_pretrain.add(Dense(output_dim))
        net_pretrain.add(Activation('softmax'))
    elif network_type == 1:
        net_pretrain.add(Conv2D(20,
              (24,1),
              # strides=(13,1),
              padding='valid',
              input_shape=(input_dim,1,1),
              data_format='channels_last'))
        net_pretrain.add(Activation('tanh'))
        net_pretrain.add(MaxPooling2D((5,1), data_format='channels_last'))
        net_pretrain.add(Flatten())
        net_pretrain.add(Dense(100))
        net_pretrain.add(Activation('tanh'))
        net_pretrain.add(Dense(output_dim))
        net_pretrain.add(Activation('softmax'))
    else:
        net_pretrain.add(Conv2D(20,
              (11,1),
              # strides=(6,1),
              padding='valid',
              input_shape=(input_dim,1,1),
              data_format='channels_last'))
        net_pretrain.add(Activation('tanh'))
        net_pretrain.add(MaxPooling2D((3,1), data_format='channels_last'))
        net_pretrain.add(Flatten())
        net_pretrain.add(Dense(100))
        net_pretrain.add(Activation('tanh'))
        net_pretrain.add(Dense(output_dim))
        net_pretrain.add(Activation('softmax'))
    
    return net_pretrain

def build_finetune(net_pretrain,shape,input_dim,output_dim,share_parameter=True):

    input_list = []
    p_list = []

    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            input_list.append(Input(shape=(input_dim,)))
    if share_parameter:
        for i in range(shape[0]*shape[1]):
            p_list.append(net_pretrain(input_list[i]))
    else:
        trained_weights = net_pretrain.get_weights()
        for i in range(shape[0]*shape[1]):
            net_pretrain_copy = clone_model(net_pretrain) # This will reinitialize the weights
            net_pretrain_copy.name = 'net_pretrain_copy_' + str(i)
            net_pretrain_copy.set_weights(trained_weights)
            p_list.append(net_pretrain_copy(input_list[i]))
            
    merged_layer = merge(p_list, mode='concat', concat_axis=-1)
    fc_layer     = Dense(100,activation='selu')(merged_layer)
    dp_layer     = Dropout(0.5)(fc_layer)
    output_layer     = Dense(output_dim, activation='softmax')(dp_layer)

    net_finetune = Model(inputs=input_list, outputs=output_layer)
    return net_finetune

