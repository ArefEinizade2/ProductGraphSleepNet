import scipy.io as sio
import random
import numpy as np
from DE_PSD import *
from tensorflow.keras.utils import to_categorical
from scipy.io import savemat
import os
import glob
#%%
def convert_labels(y):
    
    yy = np.zeros(y.shape).astype('uint8')
    
    yy[y==0] = 0
    yy[y==1] = 1
    yy[y==2] = 2
    yy[y==3] = 3
    yy[y==4] = 3
    yy[y==5] = 4

    return yy
#%%
def get_ind(A, value):
    ind = []
    for i in range(len(A)):
        if A[i] == value:
            ind.append(i)
    return ind
#%%
def get_ind_input_idxs(A, input_idxs):
    ind = []
    for j in input_idxs:
        for i in range(len(A)):
            if A[i] == j:
                ind.append(i)
    return ind

#%%
def ReadData(filename,pathF):
    '''
    Read DE or PSD from XXXX_psd_de.mat
    '''
    read_mat=sio.loadmat(pathF+filename+'_psd_de.mat')
    read_data_de=read_mat['de']
#    read_data_psd=read_mat['psd']
    return read_data_de

#%%
def ReadLabel(filename,pathL):
    '''
    Read label from XXXX-Label.mat
    '''
    read_mat=sio.loadmat(pathL+filename+'-Label.mat')
    Label_lists=read_mat['label']
    return Label_lists
#%%
def DE_PSD_a_File(FileName, shuffled_idx, all_subjects, sub, stft_para, save_dir = './', data_dir = './'):
    '''
    compute PSD and DE of a file
    '''
    n_channels = 3
    WakePick = 60
    # Read origin data from .mat files       
    
    data_x_test = []
    data_y_test = []
    
    test_ind = get_ind_input_idxs(all_subjects, [shuffled_idx[sub]])

    for in_subject in test_ind:       
        data = sio.loadmat(FileName[int(in_subject)])
        X = data['record']
        y = data['Labels']
        del data
        X = X[:n_channels]
        Trials_num = int(X.shape[-1]/3000)
        X = np.transpose(np.reshape(X, (n_channels, Trials_num, 3000)), (1, 0, 2))  
        sleep_ind = np.where(np.squeeze(y)!=0)
        sleep_ind = sleep_ind[0]
        X = X[ int(sleep_ind[0]-WakePick) : int(sleep_ind[-1]+WakePick) ]
        y = y[ int(sleep_ind[0]-WakePick) : int(sleep_ind[-1]+WakePick) ]
        data_x_test.append(np.asarray(X))  
        del X
        y = convert_labels(y)
        y = to_categorical(y, num_classes = 5)
        data_y_test.append(np.asarray(y))
    
    Data_mat = np.concatenate(data_x_test, axis=0)
    del data_x_test
    print(Data_mat.shape)
    labels_onehot = np.concatenate(data_y_test, axis=0)
    del data_y_test
    print(labels_onehot.shape)

    Data_lists=Data_mat
    print(FileName[test_ind[0]][-16:-11],Data_lists.shape,end='\t')
    data=Data_lists[0]
    print(data.shape,end='\n\t')
    
    # compute PSD\DE
    MYpsd = np.zeros([Data_lists.shape[0],Data_lists.shape[1],len(stft_para['fStart'])],dtype=float)
    MYde  = np.zeros([Data_lists.shape[0],Data_lists.shape[1],len(stft_para['fStart'])],dtype=float)
    for i in range(0,Data_lists.shape[0]):
        data=Data_lists[i]
        MYpsd[i],MYde[i]=DE_PSD(data,stft_para)
    print(MYpsd.shape,end=' ')

    # save to XXXX_psd_de.mat
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = FileName[test_ind[0]][-16:-11]
    sio.savemat(save_dir+filename+'_psd_de.mat', {'psd':MYpsd,'de':MYde})
    sio.savemat(save_dir+filename+"-Label.mat", {"label": labels_onehot})
    print('OK')
    return
#%%
def Prepare_K_Fold(Path, FileName, shuffle=False, seed=0, norm=True):
    '''
    make a set of data of 20-fold
    (to ensure independence of subjects, make every one files' data to one fold)
    --------
    Path:    include 'Path_Feature', 'Path_Label', 'Save_Name'
    shuffle: whether to disorder the file order(bool)
    seed:    random seed
    norm:    whether to normalize the data
    '''
    print('Path_Feature: ',Path['Path_Feature'])
    print('Path_Label:   ',Path['Path_Label'])
    
    # (optional) randomly scrambling data sets
    if shuffle:
        np.random.seed(seed)
        random.shuffle(FileName)

    Out_Data=[]
    Out_Label=[]
    
    Fold_Num=np.zeros([20],dtype=int)
    i = 0
    while i < 20:
        print('Fold #',int(i/1)+1,'\t',FileName[i],end=' ')
        
        FoldData = ReadData (FileName[i],Path['Path_Feature'])
        FoldLabel= ReadLabel(FileName[i],Path['Path_Label'])
        
        Fold_Num[int(i/1)]=FoldLabel.shape[0]
        
        Out_Data.append(FoldData)
        Out_Label.append(FoldLabel)
        
        print(Out_Data[int(i/1)].shape,Out_Label[int(i/1)].shape)
        
        if i==0:
            All_Data  = FoldData
            All_Label = FoldLabel
        else:
            All_Data  = np.row_stack((All_Data, FoldData))
            All_Label = np.row_stack((All_Label, FoldLabel))

        i+=1
        
    # Data standardization
    if norm:
        mean = All_Data.mean(axis=0)
        std = All_Data.std(axis=0)
        All_Data -= mean
        All_Data /= std
        for i in range(20):
            Out_Data[i] -= mean
            Out_Data[i] /= std
     
    print('All_Data:  ', All_Data.shape)
    print('All_Label: ', All_Label.shape)
    return {
        'Fold_Num':   Fold_Num,
        'Fold_Data':  Out_Data,
        'Fold_Label': Out_Label
        }
#%%
# define the path to load and save
stft_para={
    'stftn' :3000,# 30*Fs; where Fs = 100
    'fStart':[0.5, 2, 4,  6,  8, 11, 14, 22, 31],
    'fEnd'  :[4,   6, 8, 11, 14, 22, 31, 40, 50],
    'fs'    :100,
    'window':30}

Path={
    'Path_Data'   : './SleepEDF20 raw mat/',    # raw .mat files  (Already exists)
    'Path_Feature': './SleepEDF20_data/', # XXXX_psd_de.mat and XXXX-Label.mat (Will generate)
    'Save_Name'   : './SleepEDF20_DE_20Folds.npz' #  final feature extracted DE file
    }

my_folder = Path['Path_Data']    
FileName = glob.glob(my_folder + '/*.mat')
FileName2 = []
EqFalse = [24, 71, 79, 82, 127]
for i in range(len(FileName)):
    if i not in EqFalse:
        FileName2.append(FileName[i])
FileName = FileName2
del FileName2
all_subjects = []
for i in range(len(FileName)):
    all_subjects.append(int(FileName[i][-13 : -11]))
shuffled_idx = np.unique(all_subjects)

used_subs = 20 #len(shuffled_idx)
for sub in range(used_subs):                            
    print('Sub #' + str(shuffled_idx[sub]) + '/'+str(shuffled_idx[-1])+':')
    DE_PSD_a_File(FileName, shuffled_idx, all_subjects, sub, stft_para, save_dir = Path['Path_Feature'], data_dir = Path['Path_Data'])
print("DE and PSD extraction complete.")

#%%
Path={
    'Path_Data'   : './SleepEDF20 raw mat/',    # raw .mat files  (Already exists)
    'Path_Feature': './SleepEDF20_data/', # XXXX_psd_de.mat (Will generate)
    'Path_Label': './SleepEDF20_data/', # XXXX-Label.mat (Will generate)
    'Save_Name'   : './SleepEDF20_DE_20Folds.npz' # final feature extracted DE file
    }

my_folder = Path['Path_Feature']    
FileName3 = glob.glob(my_folder + '/*_psd_de.mat')
FileName4 = []
for i in range(len(FileName3)):
    FileName4.append(FileName3[i][-16:-11])
# make fold packaged data
ReadList = Prepare_K_Fold(Path, FileName4)
np.savez(
    Path['Save_Name'],
    Fold_Num   = ReadList['Fold_Num'],
    Fold_Data  = ReadList['Fold_Data'],
    Fold_Label = ReadList['Fold_Label']
    )
print('Save OK')
