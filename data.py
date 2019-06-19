# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import pickle

def load_mat(num_data_set):
    # extract data and label from mat file
    # mat file downloaded at http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

    prefix = '../../data/'
    postfix = '.mat'
    data_path = {
#            0 : 'Indian_pines_corrected',
            0 : 'Indian_pines',
            1 : 'Salinas_corrected',
            2 : 'PaviaU',
            3 : 'KSC',
            4 : 'Botswana',
            5 : 'Pavia',
            }
    gt_path = {
            # 0 : 'Indian_pines_gt',
            0 : 'Indian_pines_sel_gt',
            1 : 'Salinas_gt',
            2 : 'PaviaU_gt',
            3 : 'KSC_gt',
            4 : 'Botswana_gt',
            5 : 'Pavia_gt',
            }
    data_keys = {
#            0 : 'indian_pines_corrected',
#            0 : 'finalsub',
            0 : 'indian_pines',
            1 : 'salinas_corrected',
            2 : 'paviaU',
            3 : 'KSC',
            4 : 'Botswana',
            5 : 'pavia',
            }
    gt_keys = {
            # 0 : 'indian_pines_gt',
#            0 : 'Label0',
            0 : 'indian_pines_sel_gt',
            1 : 'salinas_gt',
            2 : 'paviaU_gt',
            3 : 'KSC_gt',
            4 : 'Botswana_gt',
            5 : 'pavia_gt',
            }

    mat_data = scipy.io.loadmat(prefix + data_path[num_data_set] + postfix)
    mat_data = mat_data[data_keys[num_data_set]]

    mat_gt = scipy.io.loadmat(prefix + gt_path[num_data_set]   + postfix)
    mat_gt = mat_gt[gt_keys[num_data_set]]
    
    return mat_data, mat_gt

def norm_data(data):
    new_data = (data - data.min()) / (data.max()- data.min()) # uint16 convers to float64
    return new_data

def remove_bad_data(data):
    data[data>10000] = 0

def expand_edge(data, r):
    '''Fill 0 at the edge of data.
    transform data from [height, width, depth]
    to [height + 2*r, width + 2*r, depth]

    # Arguments
        data: data cube, shape [height, width, depth]
        r: length of expand
    '''

    d = 2 * r
    new_data = np.zeros((data.shape[0]+d,data.shape[1]+d,data.shape[2]))
    new_data[r:-r,r:-r,:] = data

    # fill row
    new_data[0:r,r:-r] = data[r-1::-1,:]
    new_data[-r:,r:-r] = data[-1:-r-1:-1,:]
    # fill column
    new_data[r:-r,0:r] = data[:,r-1::-1]
    new_data[r:-r,-r:] = data[:,-1:-r-1:-1]
    # fill corner
    new_data[0:r,0:r] = data[r-1::-1,r-1::-1]
    new_data[0:r,-r:] = data[r-1::-1,:-r-1:-1]
    new_data[-r:,0:r] = data[:-r-1:-1,r-1::-1]
    new_data[-r:,-r:] = data[:-r-1:-1,:-r-1:-1]

    return new_data

def create_patch(data_3d,label_2d, r):
    '''Create 3D data patch corresponding to class

    # Arguments
        data_3d:
            3D data, shape [height, width, depth]
        label_2d:
            corresponding label, shape [height, width]
        r:
            patch size
    # Return
        data_patch:
            shape [2*r + 1, 2*r + 1, depth]
        label:
            corresponding label, scala
    '''
    num_labels = len(np.unique(label_2d)) - 1
    data_patch_by_class = [[] for i in range(num_labels)]

    for x in range(data_3d.shape[0]-2*r):
        for y in range(data_3d.shape[1]-2*r):
            # only use valid data
            if label_2d[x,y] != 0:
                x_min = x
                x_max = x + 2*r + 1
                y_min = y
                y_max = y + 2*r + 1
                patch = data_3d[x_min:x_max , y_min:y_max , :]
                data_patch_by_class[label_2d[x,y]-1].append(patch)
    return data_patch_by_class

def split_data(data_patch_by_class, split_method):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(data_patch_by_class)):
        num_now = len(data_patch_by_class[i])
        ns = split_method[i]
        train_x.extend(data_patch_by_class[i][0:ns])
        train_y.extend([i for j in range(ns)])
        # test_x.extend(data_patch_by_class[i][ns:])
        test_x.extend(data_patch_by_class[i][:])
        # test_y.extend([i for j in range(num_now - ns)])
        test_y.extend([i for j in range(num_now)])
    return train_x, train_y, test_x, test_y

# Leaving out classes that have less than 400 samples in the Indian_pines
def split_data_indian(data_patch_by_class, split_numbers):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    classes = 0 # class index starts from 0 and it is continuous
    for i in range(len(data_patch_by_class)):
        num_now = len(data_patch_by_class[i])
        if num_now > 400:
            train_x.extend(data_patch_by_class[i][0:split_numbers])
            train_y.extend([classes for j in range(split_numbers)])
            test_x.extend(data_patch_by_class[i][split_numbers:])
            test_y.extend([classes for j in range(num_now - split_numbers)])
            classes += 1
    return train_x, train_y, test_x, test_y

def shuffle_xy(data, label):
    sample_size = len(label)
    idx = np.arange(0,sample_size)
    np.random.shuffle(idx)
    data_shuffle = []
    label_shuffle = []
    for i in idx:
        data_shuffle.append(data[i])
        label_shuffle.append(label[i])
    return data_shuffle, label_shuffle

def patch_to_list(data):
    r = data[0].shape[0]
    data_list = [np.zeros((len(data),data[0].shape[2])) for i in range(r*r)]
    for idx,d in enumerate(data):
        for i in range(r):
            for j in range(r):
                data_list[i*r+j][idx] = d[i,j,:]
    return data_list

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(len(np.unique(x)))[x]

def create_data_set(num_data_set, num_range, split_method):
    data_3d,label_2d    = load_mat(num_data_set)
    remove_bad_data(data_3d)
    data_3d             = norm_data(data_3d)
    data_3d             = expand_edge(data_3d, num_range)
    data_patch_by_class = create_patch(data_3d, label_2d, num_range)

    for i in range(len(data_patch_by_class)):
        np.random.shuffle(data_patch_by_class[i])

    (train_x,
    train_y,
    test_x,
    test_y)             = split_data(data_patch_by_class, split_method)
    
    # Indian Pines 9 classes: 
    # (train_x,
    # train_y,
    # test_x,
    # test_y)             = split_data_indian(data_patch_by_class, split_method)

    train_x, train_y    = shuffle_xy(train_x,train_y)
    test_x, test_y      = shuffle_xy(test_x,test_y)

    train_x_1d          = np.array(train_x)[:,num_range,num_range,:]
    test_x_1d           = np.array(test_x)[:,num_range,num_range,:]
    train_x_1d          = train_x_1d.reshape((train_x_1d.shape[0],train_x_1d.shape[1],1,1))
    test_x_1d           = test_x_1d.reshape((test_x_1d.shape[0],test_x_1d.shape[1],1,1))

    train_x_1d = train_x_1d[:,:,0,0] #redundant???
    test_x_1d  = test_x_1d[:,:,0,0]
    
    train_x_3d          = patch_to_list(train_x)
    test_x_3d           = patch_to_list(test_x)

    train_y_onehot      = one_hot_encode(train_y)
    test_y_onehot       = one_hot_encode(test_y)

    return train_x_3d, train_x_1d, train_y, train_y_onehot, test_x_3d, test_x_1d, test_y, test_y_onehot
