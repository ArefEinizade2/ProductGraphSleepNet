import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import shutil

import keras
import tensorflow as tf

import keras.backend.tensorflow_backend as KTF

from ProductGraphSleepNet import *
from SleepEDF20_Utils import *
from SleepEDF20_DataGenerator import *
from tensorflow.keras.utils import plot_model
from keras.models import Model
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from random import shuffle
#%%
# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file")
parser.add_argument("-g", type = str, help = "GPU number to use, set '-1' to use CPU")
args = parser.parse_args(["-c", "./SleepEDF20.config", "-g", "0"])
Path, cfgTrain, cfgModel = ReadConfig(args.c)

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
if args.g != "-1":
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print("Use GPU #"+args.g)
else:
    print("Use CPU only")

# ## 1.2. Analytic parameters

# [train] parameters
use_pretrained = cfgTrain["use_pretrained"]
channels   = int(cfgTrain["channels"])
fold       = int(cfgTrain["fold"])
context    = int(cfgTrain["context"]) # P (number of neighbor sleep epochs) in the paper
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer  = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])
lr_decay   = float(cfgTrain["lr_decay"])

# [model] hyperparameters
GLalpha               = float(cfgModel["GLalpha"])# lambda in the paper
num_of_chev_filters   = int(cfgModel["cheb_filters"]) # F' in the paper
cheb_k                = int(cfgModel["cheb_k"]) # K in the paper
l1                    = float(cfgModel["l1"])
l2                    = float(cfgModel["l2"])
dropout               = float(cfgModel["dropout"])
GRU_Cell               = int(cfgModel["GRU_Cell"]) # beta in the paper
attn_heads               = int(cfgModel["attn_heads"]) # H in the paper

## 1.3. Parameter check and enable

# check optimizer（opt）
if optimizer=="adam":
    opt = keras.optimizers.Adam(lr=learn_rate,decay=lr_decay)
elif optimizer=="RMSprop":
    opt = keras.optimizers.RMSprop(lr=learn_rate,decay=lr_decay)
elif optimizer=="SGD":
    opt = keras.optimizers.SGD(lr=learn_rate,decay=lr_decay)
else:
    assert False,'Config: check optimizer'

# set l1、l2（regularizer）
if l1!=0 and l2!=0:
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
elif l1!=0 and l2==0:
    regularizer = keras.regularizers.l1(l1)
elif l1==0 and l2!=0:
    regularizer = keras.regularizers.l2(l2)
else:
    regularizer = None
    
# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save']+"SleepEDF20_last.config")

#%%
# 2. Read data and process data

## 2.1. Read data
ReadList = np.load(Path['data'], allow_pickle=True)


Fold_Num    = ReadList['Fold_Num']     # Samples of each fold
# Out: list with same length
Fold_Data    = ReadList['Fold_Data']
Fold_Label   = ReadList['Fold_Label']
print("Read data successfully")

## 2.2. Add time context

Fold_Data    = AddContext(Fold_Data,context)
Fold_Label   = AddContext(Fold_Label,context,label=True)
Fold_Num_c  = Fold_Num + 1 - context
        
print('Context added successfully.')
print('Number of samples: ',np.sum(Fold_Num_c))

DataGenerator = kFoldGenerator(fold,Fold_Data,Fold_Label)

#%%
# 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
LearnedGraphsSpatial = []
LearnedGraphsTempral = []
predicts_learnedSpatialGraphs_mean_All_folds = []
predicts_learnedTemporalGraphs_mean_All_folds = []
predicts_learnedSpatialGraphs_All_folds = []
predicts_learnedTemporalGraphs_All_folds = []

for i in range(fold):

    opt = keras.optimizers.Adam(lr=learn_rate,decay=lr_decay)
    
    print(128*'>')
    print('Fold ' + str(i+1) + '/' + str(fold))
    
    # get i th-fold data
    train_data, train_targets, val_data, val_targets, test_data, test_targets = DataGenerator.getFold(i)
    sample_shape = (context,train_data.shape[2],train_data.shape[3])   
    
    # build model
    model = build_ProductGraphSleepNet(k=cheb_k, num_of_chev_filters=num_of_chev_filters, 
                      sample_shape=sample_shape, opt=opt, GLalpha=GLalpha, 
                      regularizer=regularizer, GRU_Cell=GRU_Cell, attn_heads=attn_heads, dropout=dropout)  

    if i==0:
        model.summary()
        plot_model(model, to_file=str(Path['Save'] + 'SleepEDF20_model.png'), show_shapes=True)


    if use_pretrained=='False': # if pretrain is selected, the model is not trained
        # train
        history = model.fit(
            x = train_data,
            y = train_targets,
            epochs = num_epochs,
            batch_size = batch_size,
            shuffle = True, verbose = 2,
            validation_data = (val_data, val_targets),
            callbacks=[keras.callbacks.ModelCheckpoint(Path['Save']+'SleepEDF20_Best_model_'+str(i)+'.h5', 
                                                       monitor='val_acc', 
                                                       verbose=2, 
                                                       save_best_only=True, 
                                                       save_weights_only=False, 
                                                       mode='auto', 
                                                       period=1 )])
        # Load weights of best performance
        model.load_weights(Path['Save']+'SleepEDF20_Best_model_'+str(i)+'.h5')
    else:
        # Load weights of best performance
        model.load_weights(Path['pretrain_folder']+'SleepEDF20_Best_model_'+str(i)+'.h5')
          
    # Test on the unseen subjects
    test_mse, test_acc = model.evaluate(test_data, test_targets, verbose=0)
    print('Evaluate', np.round(test_acc, 3))
    all_scores.append(np.round(test_acc, 3))
        
    print('acc of folds till now: ')
    print(all_scores)
    print('mean of the acc of folds till now: ')
    print(np.round(np.mean(all_scores), 3))    
    print(128*'>')
    
    # Predict
    predicts = model.predict(test_data)

    # Learned graphs Predict 
    model_JustGL_Spatial = Model(inputs = model.input, outputs = model.get_layer(index=2).output)
    predicts_learnedGraphs = model_JustGL_Spatial.predict(test_data)
    predicts_learnedSpatialGraphs_All_folds.append(np.array(predicts_learnedGraphs))

    model_JustGL_Temporal = Model(inputs = model.input, outputs = model.get_layer(index=6).output)
    predicts_learnedGraphs = model_JustGL_Temporal.predict(test_data)
    predicts_learnedTemporalGraphs_All_folds.append(np.array(predicts_learnedGraphs))

    # concatenation of all predicted labels:
    AllPred_temp = np.argmax(predicts, axis=1)
    if i == 0:
        AllPred = AllPred_temp
        AllOneHot_true = test_targets
        AllOneHot_pred = predicts
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllOneHot_true = np.concatenate((AllOneHot_true, test_targets), axis = 0)
        AllOneHot_pred = np.concatenate((AllOneHot_pred, predicts), axis = 0)

    # Fold finish
    print(128*'_')
    del model,train_data,train_targets,val_data,val_targets, test_data, test_targets

    KTF.clear_session()

#%%   
# 4. Final results:

# save learned graphs:
np.save(Path['Save'] + 'SleepEDF20_LearnedGraphsSpatial.npy', np.concatenate(predicts_learnedSpatialGraphs_All_folds, axis=0))
np.save(Path['Save'] + 'SleepEDF20_LearnedGraphsTempral.npy', np.concatenate(predicts_learnedTemporalGraphs_All_folds, axis=0))

# print acc of each fold
print(128*'=')
print("All folds' acc: ",all_scores)
print("Average acc of each fold: ",np.mean(all_scores))

# Get all true labels
AllTrue = np.argmax(AllOneHot_true, axis=1)
np.save(Path['Save'] + 'SleepEDF20_Labels_All.npy', AllTrue)


# Print score to console
print(128*'=')
PrintScore(AllTrue, AllPred)
# Print score to Result.txt file
PrintScore2(AllTrue, AllPred, all_scores, savePath=Path['Save']+'SleepEDF20_')

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W','N1','N2','N3','REM'], savePath=Path['Save']+'SleepEDF20_', title='Confusion matrix SleepEDF20')

# AUC/AUPRC plots:
class_names = ['W','N1','N2','N3','R']
y_test = AllOneHot_true
y_score = AllOneHot_pred
lw = 2
n_classes = 5
fpr = dict()
tpr = dict()
roc_auc = dict()
precision = dict()
recall = dict()
roc_auprc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    roc_auprc[i] = average_precision_score(y_test[:, i], y_score[:, i])
colors = ['b', 'c', 'y', 'm', 'r']
plt.figure(figsize=(16, 8))
#plt.rcParams.update({'font.size': 10, 'font.weight': 'normal'})
plt.subplot(1,2,1)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='Class ' + class_names[i] +' (area = ' + str(round(roc_auc[i], 2)) + ')')    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC plot')
plt.legend(loc="lower right")

plt.subplot(1,2,2)
for i, color in zip(range(n_classes), colors):
#        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Class ' + class_names[i] +' (area = ' + str(round(roc_auprc[i], 2)) + ')')    
    
plt.plot([1, 0], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.title("AUPRC plot")
plt.show()
plt.savefig(Path['Save'] + 'SleepEDF20_AUC_AUPRC_plot.eps')   
plt.savefig(Path['Save'] + 'SleepEDF20_AUC_AUPRC_plot.png')   
plt.savefig(Path['Save'] + 'SleepEDF20_AUC_AUPRC_plot.pdf')   


