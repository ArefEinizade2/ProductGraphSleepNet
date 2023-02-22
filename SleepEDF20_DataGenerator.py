import numpy as np
import random

val_list = [16, 17,  7, 11,  6,  9,  0, 13, 14,  3, 18, 19,  5,  4,  8, 10, 12,
       15,  1,  2]

class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, k, x, y):
        if len(x)!=k or len(y)!=k:
            assert False,'Data generator: Length of x or y is not equal to k.'
        self.k=k
        self.x_list=x
        self.y_list=y

    # Get i-th fold
    def getFold(self, i):
        
        train_data = []; train_targets = []; val_data = []; val_targets = []; test_data = []; test_targets = [];
        channel_set = set(np.arange(0, 3))
        channel_set = list(channel_set)
        
        isFirst=True
        isFirst_val=True
        
        train_set = set(np.arange(0,self.k))
        train_set = train_set.difference({i})
        train_set = train_set.difference({val_list[i]})
        val_idx = [val_list[i]]
        print('Validation fold: '+ str(list(np.array(val_idx)+1)))
        
        for p in range(self.k): 
            if p!=i and p not in val_idx:
                if isFirst:
                    train_data       = self.x_list[p][:, :, channel_set]
                    train_targets    = self.y_list[p]
                    isFirst = False
                else:
                    train_data      = np.concatenate((train_data, self.x_list[p][:, :, channel_set]))
                    train_targets   = np.concatenate((train_targets, self.y_list[p]))
            elif p in val_idx:
                if isFirst_val:
                    val_data    = self.x_list[p][:, :, channel_set]
                    val_targets = self.y_list[p]
                    isFirst_val = False
                else:
                    val_data      = np.concatenate((val_data, self.x_list[p][:, :, channel_set]))
                    val_targets   = np.concatenate((val_targets, self.y_list[p]))
            else:
                test_data    = self.x_list[p][:, :, channel_set]
                test_targets = self.y_list[p]

        return train_data, train_targets, val_data, val_targets, test_data, test_targets

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1,self.k):
            All_X = np.append(All_X,self.x_list[i], axis=0)
        return All_X

    # Get all label y
    def getY(self):
        All_Y = self.y_list[0]
        for i in range(1,self.k):
            All_Y = np.append(All_Y,self.y_list[i], axis=0)
        return All_Y

    # Get all label y
    def getY_one_hot(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)