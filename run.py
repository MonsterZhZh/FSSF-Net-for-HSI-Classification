#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:09:09 2017

@author: zpl
"""
# Set to CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import sys
import pickle
import time
from data import create_data_set
from net import build_pretrain, build_pretrain_conv, build_finetune, build_pretrain_local, build_pretrain_cnn
from keras import optimizers
from evaluate import res2excel
import keras.backend as K
import time


if __name__ == '__main__':

    CONFIG = {
             # Dataset, split_each_class, N_RANGE->N_WINDOW, N_REPEAT, FREEZE, SHARE, PRE-TRAIN,
             # PRE_TRAIN_NET_TYPE, HIDDEN(units)
             # indian 5%
             # for finalsub: [95,489,834,968,54,614,497,1294,380,26,234,20,1434,2468,747,212]
#            '0' : [0,[5,24,42,48,3,31,25,65,19,1,12,1,72,123,37,11],3,5,False,True,True,0,100], 
             # for gic data: [46,1428,830,237,483,730,28,478,20,972,2455,593,205,1265,386,93]
#            '0' : [0,[2,71,42,12,24,36,1,24,1,48,123,30,10,63,19,5],3,5,False,True,True,0,100],
#            '1' : [0,[2,71,42,12,24,36,1,24,1,48,123,30,10,63,19,5],3,5,False,False,True,0],
#            '2' : [0,[2,71,42,12,24,36,1,24,1,48,123,30,10,63,19,5],3,5,False,True,False,0],
#            '3' : [0,[2,71,42,12,24,36,1,24,1,48,123,30,10,63,19,5],3,5,True,True,True,0],
#            # indian 10%
#            '4' : [0,[5,143,83,24,48,73,3,48,2,97,246,59,21,127,39,9],3,5,False,True,True,0,100],
#            '5' : [0,[5,143,83,24,48,73,3,48,2,97,246,59,21,127,39,9],3,5,False,False,True,0],
#            '6' : [0,[5,143,83,24,48,73,3,48,2,97,246,59,21,127,39,9],3,5,False,True,False,0],
#            '7' : [0,[5,143,83,24,48,73,3,48,2,97,246,59,21,127,39,9],3,5,True,True,True,0],
#            # pavia university 1%
#            '8' : [2,[5,5,4,5,3,5,4,5,2],3,2,False,True,True,0,100],
#            '9' : [2,[5,5,4,5,3,5,4,5,2],3,5,False,False,True,0],
#            '10' : [2,[5,5,4,5,3,5,4,5,2],3,5,False,True,False,0],
#            '11' : [2,[5,5,4,5,3,5,4,5,2],3,5,True,True,True,0],
#            # pavia university 10%
#            '12' : [2,[55,54,39,52,27,53,38,51,23],3,5,False,True,True,0,100],
#            '13' : [2,[55,54,39,52,27,53,38,51,23],3,5,False,False,True,0],
#            '14' : [2,[55,54,39,52,27,53,38,51,23],3,5,False,True,False,0],
#            '15' : [2,[55,54,39,52,27,53,38,51,23],3,5,True,True,True,0],
#            # pavia university 50, different pretrain net
            # '16' : [2,[50 for i in range(9)],3,2,False,True,True,0,100],
            '17' : [2,[50 for i in range(9)],3,5,False,True,True,1,100],
#            '18' : [2,[50 for i in range(9)],3,5,False,True,True,2,100],
#            # pavia university 200
           # '19' : [2,[200 for i in range(9)],3,5,False,True,True,0,100],
#            # Salinas 
            '20' : [1,[50 for i in range(16)],3,5,False,True,True,1,100],
#            '21' : [1,[68 for i in range(16)],3,5,False,True,True,0,100],
#            '22' : [1,[100 for i in range(16)],3,5,False,True,True,0,100],
           # '23' : [1,[200 for i in range(16)],3,5,False,True,True,0,100],
#            # Botswana 
#            '24' : [4,[50 for i in range(14)],3,5,False,True,True,0,100],
#            '25' : [4,[100 for i in range(14)],3,5,False,True,True,0,100],
#            '26' : [4,[200 for i in range(14)],3,5,False,True,True,0,100],
            # hidden
#            '30' : [2,[50 for i in range(14)],3,5,False,True,True,0,40],
#            '31' : [2,[50 for i in range(14)],3,5,False,True,True,0,70],
#            '32' : [2,[50 for i in range(14)],3,5,False,True,True,0,100],
#            '33' : [2,[50 for i in range(14)],3,5,False,True,True,0,130],
#            '34' : [2,[50 for i in range(14)],3,5,False,True,True,0,160],
            # Salinas 5%
#            '35' : [1,[100,186,99,70,134,198,179,564,310,164,53,96,46,54,363,90],3,5,False,True,True,0,100],
            # Salinas 10%
#            '36' : [1,[201,373,198,139,268,396,358,1127,620,328,107,193,92,107,727,181],3,5,False,True,True,0,100],
            # Salinas 25%
#            '37' : [1,[502,932,494,348,670,990,895,2818,1551,820,267,482,229,268,1817,452],3,5,False,True,True,0,100],
            # Indian_pines_220Bands 25%:
#            '38' : [0,[12,357,208,59,121,183,7,120,5,243,614,148,51,316,97,23],3,5,False,True,True,0,100],
            # Indian_pines_220Bands 200:
            # '39' : [0,[200 for i in range(9)],3,5,False,True,True,0,100],
            # Pavia Center 200:
            # '40' : [5,[200 for i in range(9)],3,5,False,True,True,0,100],
            # Indian_pines_220Bands 50
            '41' : [0,[50 for i in range(9)],3,5,False,True,True,1,100],
            # Indian Pines(220 Bands still gic dat) 60%
            # '42' : [0,[27,856,498,142,289,438,16,286,12,583,1473,355,123,759,231,55],3,5,False,True,True,0,100],
            # Salinas 60%
            # '43' : [1,[1206,2238,1188,834,1608,2376,2148,6762,3720,1968,642,1158,552,642,4362,1086],3,5,False,True,True,0,100],
            # Pavia University 60%
            # '44' : [2,[330,324,234,312,162,318,228,306,138],3,5,False,True,True,0,100],
            # KSC(761, 243, 256, 252, 161, 229, 105, 431, 520, 404, 419, 503, 927) 50 MLP versus CNN
            # '45' : [3,[50 for i in range(13)],3,5,False,True,True,0,100],
            '46' : [3,[50 for i in range(13)],3,5,False,True,True,1,100],
            # '47' : [3,[456,145,153,151,96,137,63,258,312,242,251,301,556],3,5,False,True,True,0,100],
            }
    # file = open('time.txt','w')
    for config_name, config in CONFIG.items():
        # DATA_SET, 0 Indian pines, 1 Salinas, 2 Pavia University, 3 KSC, 4 Botswana
        DATA_SET = config[0]
        # SPLIT is a list of int, represent training pixel number of each class
        # KSC [761,243,256,252,161,229,105,431,520,404,419,503,927]
        # list index is class label, 0~8 for Pavia University
        # SPLIT = [200 for i in range(9)]
        SPLIT = config[1]
        # 7  x  7 window, N_RANGE = 3, N_WINDOW=7
        # 11 x 11 window, N_RANGE = 5, N_WINDOW=11
        N_RANGE = config[2]
        N_WINDOW = N_RANGE * 2 + 1
        # repeat experiment
        N_REPEAT = config[3]
        # freeze pretrain net's parameter while finetune
        FREEZE = config[4]
        # share parameter
        SHARE = config[5]
        # with pretrain
        PRETRAIN = config[6]
        # pretrain net type, 0 full connect, 1 1D-CNN, 2 local connect
        PRETRAIN_NET_TYPE = config[7]
        HIDDEN = config[8]
        
        # SPLIT = [2,71,42,12,24,36,1,24,1,48,123,30,10,63,19,5]
        # SPLIT = [5,143,83,24,48,73,3,48,2,97,246,59,21,127,39,9]
        # SPLIT = [30,150,150,100,150,150,20,150,15,150,150,150,150,150,50,50]
        # SPLIT = [548,540,392,524,265,532,375,514,231]
        # SPLIT = [5,5,4,5,3,5,4,5,2]
        # SPLIT = [55,54,39,52,27,53,38,51,23]
        # SPLIT = [67,67,67,69,67,67,68,69,68,68,68,67,67,67,70,67]
    
        # parameter for train
        LR_INIT_PRETRAIN = 0.001
        LR_INIT_FINETUNE = 0.001
        # LR_INIT_FINETUNE = 0.0001 # adjusted for Salinas
        LR_DECAY_PRETRAIN = 5e-3
        LR_DECAY_FINETUNE = 1e-2
        # LR_DECAY_FINETUNE = 1e-3 # adjusted for Salinas
        EPOCH_PRETRAIN = 10000
        EPOCH_FINETUNE = 1000
        # EPOCH_FINETUNE = 3000 # adjusted for Salinas
    
        # result list
        RES_PREDICT_PRETRAIN = []
        RES_PREDICT_FINETUNE = []
        RES_GT = []
        
        for i in range(N_REPEAT):
        
            # load data
            (train_x_3d,
             train_x_1d,
             train_y,
             train_y_onehot,
             test_x_3d,
             test_x_1d,
             test_y,
             test_y_onehot) = create_data_set(DATA_SET, N_RANGE, SPLIT)
            # start  = time.time()
            # pretrain
            if PRETRAIN_NET_TYPE == 0:
                net_pretrain = build_pretrain(train_x_1d.shape[1],
                                              train_y_onehot.shape[1],
                                              HIDDEN)
            elif PRETRAIN_NET_TYPE == 1:
                net_pretrain = build_pretrain_conv(train_x_1d.shape[1],
                                              train_y_onehot.shape[1])
                # net_pretrain = build_pretrain_cnn(train_x_1d.shape[1], train_y_onehot.shape[1], DATA_SET)
            elif PRETRAIN_NET_TYPE == 2:
                net_pretrain = build_pretrain_local(train_x_1d.shape[1],
                                              train_y_onehot.shape[1])
            if PRETRAIN:
                opt = optimizers.Adam(lr=LR_INIT_PRETRAIN,
                                      beta_1=0.9,
                                      beta_2=0.999,
                                      epsilon=1e-08,
                                      decay=LR_DECAY_PRETRAIN)
                net_pretrain.compile(optimizer=opt,
                                     loss='categorical_crossentropy',
                                     metrics=['categorical_accuracy'])
                net_pretrain.fit(train_x_1d,
                       train_y_onehot,
        #               validation_data=(test_x_1d,test_y_onehot),
                       epochs=EPOCH_PRETRAIN,
                       batch_size=train_x_1d.shape[0],
                       shuffle=True,
                       verbose=1)
            # save pretrain result
            RES_PREDICT_PRETRAIN.append(net_pretrain.predict(test_x_1d))
            
            # finetune
            if FREEZE:
                for layer in net_pretrain.layers:
                    layer.trainable = False
                
            net_finetune = build_finetune(net_pretrain,
                                          [N_WINDOW,N_WINDOW],
                                          test_x_1d.shape[1],
                                          test_y_onehot.shape[1],
                                          share_parameter=SHARE)
            opt = optimizers.Adam(lr=LR_INIT_FINETUNE,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-08,
                                  decay=LR_DECAY_FINETUNE)
            net_finetune.compile(optimizer=opt,
                                 loss='categorical_crossentropy',
                                 metrics=['categorical_accuracy'])
            net_finetune.fit(train_x_3d, train_y_onehot,
    #                    validation_data=(test_x_3d,test_y_onehot),
                        epochs=EPOCH_FINETUNE,
                        # batch_size=int(train_x_3d[0].shape[0]/3),
                        batch_size=train_x_3d[0].shape[0],
                        shuffle=True,
                        verbose=1)
            # train_time = time.time() - start
            # file.write('dataset:')
            # file.write(str(config[0]))
            # file.write(',iter:')
            # file.write(str(i))
            # file.write(',train time:')
            # file.write(str(train_time))
            # start  = time.time()
            # save finetune result
            RES_PREDICT_FINETUNE.append(net_finetune.predict(test_x_3d))
            RES_GT.append(test_y)
            # test_time = time.time() - start
            # file.write(',test time:')
            # file.write(str(test_time))
            # file.write('\n')
            # clear session for next loop
            # del train_x_3d, train_y, train_y_onehot, test_x_3d, test_y, test_y_onehot
            K.clear_session()
            
        # save result to npz file
        np.savez('../result/res_{0:s}.npz'.format(config_name),
                 RES_PREDICT_PRETRAIN = RES_PREDICT_PRETRAIN,
                 RES_PREDICT_FINETUNE = RES_PREDICT_FINETUNE,
                 RES_GT = RES_GT)
        
        # save finetune result to excel
        res2excel(config_name)
    # file.close()