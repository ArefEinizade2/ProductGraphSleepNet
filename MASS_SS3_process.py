import scipy.io as sio
import random
import numpy as np
from DE_PSD import *
import mne
from tensorflow.keras.utils import to_categorical
from scipy.io import savemat
import os
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
def DE_PSD_a_File(filename, stft_para, save_dir='./MASS_SS3_data/', data_dir='./SS3 raw edf/'):
    '''
    compute PSD and DE of a file
    '''
    # Read origin data from raw.edf files
    raw_train = mne.io.read_raw_edf(data_dir+filename+' PSG.edf', preload=True)
            
    annot_train = mne.read_annotations(data_dir+filename+' Base.edf')
    raw_train.set_annotations(annot_train, emit_warning=False)
    
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
    
    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                     annot_train[-2]['onset'] + 30 * 60)
    raw_train.set_annotations(annot_train, emit_warning=False)

    events_train, _ = mne.events_from_annotations(raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)
    
    # create a new event_id that unifies stages 3 and 4
    if int(filename[-2:]) != 36:
        event_id = {'Sleep stage W': 1,
                    'Sleep stage 1': 2,
                    'Sleep stage 2': 3,
                    'Sleep stage 3/4': 4,
                    'Sleep stage R': 5}
    else:
        event_id = {'Sleep stage W': 1,
                    'Sleep stage 1': 2,
                    'Sleep stage 2': 3,
                    'Sleep stage R': 5}
            
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
    
    # in the following lines the filtering is performed which we didn't use them:
#    Channels = raw_train.info['ch_names'][:27]    
#    EEG_channels = []
#    EMG_channels = []
#    ECG_channels = []
#    EOG_channels = []    
#    for ch in range(len(Channels)):
#        if 'EEG' in Channels[ch]:
#            EEG_channels.append(Channels[ch])
#        elif 'EMG' in Channels[ch]:
#            EMG_channels.append(Channels[ch])
#        elif 'EOG' in Channels[ch]:
#            EOG_channels.append(Channels[ch])
#        elif 'ECG' in Channels[ch]:
#            ECG_channels.append(Channels[ch])
#            
#    raw_train.filter( l_freq=0.3, h_freq=100, 
#                                 picks=EEG_channels)
#    raw_train.filter( l_freq=0.1,  h_freq=100, 
#                                 picks=EOG_channels)
#    raw_train.filter( l_freq=0.1,  h_freq=100, 
#                                 picks=ECG_channels)
#    raw_train.filter( l_freq=10,  h_freq=100, 
#                                 picks=EMG_channels)

    epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    
    print(epochs_train)
    
    Data_mat = epochs_train.get_data()
    labels = epochs_train.events[:, 2]
    labels_onehot = to_categorical(labels-1, num_classes=5)


    Data_lists=Data_mat[:, :27,:]
    print(filename,Data_lists.shape,end='\t')
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
    sio.savemat(save_dir+filename+'_psd_de.mat', {'psd':MYpsd,'de':MYde})
    sio.savemat(save_dir+filename+"-Label.mat", {"label": labels_onehot})
    print('OK')
    return
#%%
def Prepare_K_Fold(Path, FileName, shuffle=False, seed=0, norm=True):
    '''
    make a set of data of 16-fold
    (to ensure independence of subjects, make every four files' data to one fold)
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
    
    Fold_Num=np.zeros([16],dtype=int)
    i = 0
    while i < 60:
        print('Fold #',int(i/4)+1,'\t',FileName[i],end=' ')
        
        FoldData = ReadData (FileName[i],Path['Path_Feature'])
        FoldLabel= ReadLabel(FileName[i],Path['Path_Label'])
        
        print(' ',FileName[i+1],end='  ')
        
        FoldData = np.row_stack((FoldData, ReadData (FileName[i+1],Path['Path_Feature'])))
        FoldLabel= np.row_stack((FoldLabel,ReadLabel(FileName[i+1],Path['Path_Label'])))
        
        print(' ',FileName[i+2],end='  ')

        FoldData = np.row_stack((FoldData, ReadData (FileName[i+2],Path['Path_Feature'])))
        FoldLabel= np.row_stack((FoldLabel,ReadLabel(FileName[i+2],Path['Path_Label'])))
        
        print(' ',FileName[i+3],end='  ')

        FoldData = np.row_stack((FoldData, ReadData (FileName[i+3],Path['Path_Feature'])))
        FoldLabel= np.row_stack((FoldLabel,ReadLabel(FileName[i+3],Path['Path_Label'])))

        Fold_Num[int(i/4)]=FoldLabel.shape[0]
        
        Out_Data.append(FoldData)
        Out_Label.append(FoldLabel)
        
        print(Out_Data[int(i/4)].shape, Out_Label[int(i/4)].shape)
        
        if i==0:
            All_Data  = FoldData
            All_Label = FoldLabel
        else:
            All_Data  = np.row_stack((All_Data, FoldData))
            All_Label = np.row_stack((All_Label, FoldLabel))

        i+=4
        
    i = 60
    print('Fold #',int(i/4)+1,'\t',FileName[i],end=' ')
    
    FoldData = ReadData (FileName[i],Path['Path_Feature'])
    FoldLabel= ReadLabel(FileName[i],Path['Path_Label'])
    
    print(' ',FileName[i+1],end='  ')
    
    FoldData = np.row_stack((FoldData, ReadData (FileName[i+1],Path['Path_Feature'])))
    FoldLabel= np.row_stack((FoldLabel,ReadLabel(FileName[i+1],Path['Path_Label'])))
    
    Fold_Num[int(i/4)]=FoldLabel.shape[0]
    
    Out_Data.append(FoldData)
    Out_Label.append(FoldLabel)
    
    print(Out_Data[int(i/4)].shape,Out_Label[int(i/4)].shape)
    
    All_Data  = np.row_stack((All_Data, FoldData))
    All_Label = np.row_stack((All_Label, FoldLabel))

    # Data standardization
    if norm:
        mean = All_Data.mean(axis=0)
        std = All_Data.std(axis=0)
        All_Data -= mean
        All_Data /= std
        for i in range(16):
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
Path={
    'Path_Data'   : './MASS SS3 raw edf/',    # raw .edf files  (Already exists)
    'Path_Label'  : './MASS_SS3_data/',    # XXXX-Label.mat(Will generate)
    'Path_Feature': './MASS_SS3_data/', # XXXX_psd_de.mat(Will generate)
    'Save_Name'   : './MASS_SS3_DE_16Folds.npz' # final feature extracted DE file
}
# the parameters to extract DE and PSD
stft_para={
    'stftn' :7680, # 30*Fs; where Fs = 256
    'fStart':[0.5, 2, 4,  6,  8, 11, 14, 22, 31],
    'fEnd'  :[4,   6, 8, 11, 14, 22, 31, 40, 50],
    'fs'    :256,
    'window':30,
}
# the file No of the MASS SS3
FileName=['01-03-0001', '01-03-0002', '01-03-0003', '01-03-0004', '01-03-0005', '01-03-0006', 
          '01-03-0007', '01-03-0008', '01-03-0009', '01-03-0010', '01-03-0011', '01-03-0012', 
          '01-03-0013', '01-03-0014', '01-03-0015', '01-03-0016', '01-03-0017', '01-03-0018',
          '01-03-0019', '01-03-0020', '01-03-0021', '01-03-0022', '01-03-0023', '01-03-0024', 
          '01-03-0025', '01-03-0026', '01-03-0027', '01-03-0028', '01-03-0029', '01-03-0030', 
          '01-03-0031', '01-03-0032', '01-03-0033', '01-03-0034', '01-03-0035', 
          '01-03-0037', '01-03-0038', '01-03-0039', '01-03-0040', '01-03-0041', '01-03-0042', 
          '01-03-0044', '01-03-0045', '01-03-0046', '01-03-0047', '01-03-0050', '01-03-0051', 
          '01-03-0052', '01-03-0053', '01-03-0054', '01-03-0055', '01-03-0056', '01-03-0057', 
          '01-03-0058', '01-03-0059', '01-03-0060', '01-03-0061', '01-03-0062', '01-03-0063',
          '01-03-0064', '01-03-0036', '01-03-0048'] 

FileName512=['01-03-0040', '01-03-0045', '01-03-0047', '01-03-0050', '01-03-0051', 
          '01-03-0052', '01-03-0053', '01-03-0054', '01-03-0055', '01-03-0056', '01-03-0057', 
          '01-03-0058', '01-03-0059', '01-03-0060', '01-03-0061', '01-03-0062', '01-03-0063',
          '01-03-0064', '01-03-0048'] 

#%% DE extraction and save:
for file in FileName:        
    if file in FileName512:
        stft_para['fs'] = 512
    else:
        stft_para['fs'] = 256
                    
    print(file)
    DE_PSD_a_File(file, stft_para, Path['Path_Feature'], Path['Path_Data'])
print("DE and PSD extraction complete.")
#%% # make fold packaged data

ReadList = Prepare_K_Fold(Path, FileName)
np.savez(
    Path['Save_Name'],
    Fold_Num   = ReadList['Fold_Num'],
    Fold_Data  = ReadList['Fold_Data'],
    Fold_Label = ReadList['Fold_Label']
    )
print('Save OK')
