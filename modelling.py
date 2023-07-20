# Imports
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from wrangler import preprocess
from wranglergoofy import preprocess as pp2
import optuna
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model import RNNmodel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader
import random
torch.manual_seed(47)
import pickle as pkl
ver = 'regions'
region='MOs'

if ver == 'regions':
    testrundata = preprocess(10,ver=ver,reg=region)
    feats = testrundata['features']
    labels = testrundata['labels']
    mins = testrundata['cutoffs']

    
    X = feats
    y = labels

    #subsampling
    # for en,sample in enumerate(X):
        
    #     indexsample = np.random.choice(range(sample.shape[0]),size=mins[region])
    #     X[en] = sample[indexsample]

elif ver == 'single-session':
    testrundata = preprocess(10,ver=ver)
    feats = testrundata['features']
    labels = testrundata['labels']
    X = feats['Cori2016-12-14']
    y = labels['Cori2016-12-14']
elif ver == 'goofy':
    data = pp2()
    X = data['features']
    y = data['labels']
    
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)



def optim(trial):
    # Set the hyperparameters
    cutv = trial.suggest_categorical('cut',[10,20,30,40,50,60,80,100])
    X_t = [v[:,-cutv:] for v in X if len(v)>cutv]
    X_t = [np.concatenate([np.zeros([v.shape[0],cutv-v.shape[1]]),v],axis=1) for v in X_t]
    #X_train, X_test, y_train,  y_test = train_test_split(X_t,y,test_size=.30)
    input_size = 75  # Dimensionality of the input features
    hidden_size = trial.suggest_categorical('hidden_size',[32,64,128,256,512]) #trial.suggest_categorical('hidden_size',[32,64,128,256])  # Number of units in the hidden layer
    num_layers = trial.suggest_categorical('num_layers',[1,2,3])  # Number of recurrent layers
    model_arch = trial.suggest_categorical('model_arch',['lstm','gru','rnn'])
    num_epochs = 1000
    batch_size = 250
    dropout = trial.suggest_categorical('dropout',[0,0.1,0.2])
    l2 = trial.suggest_categorical('l2',[0,0.0001,0.001,0.01])
    lr = trial.suggest_categorical('lr',[0.000001,0.00001,0.0001])
    output_size = 1  # Number of classes (binary)
    model = RNNmodel(input_size,hidden_size,num_layers,output_size,model_arch,num_epochs,batch_size,dropout,l2,lr)
    
    return model.cross_validate(X_t,y)
    
    
    
study_name = "flatbrain"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name,direction="maximize",study_name=study_name,load_if_exists=True)
study.optimize(optim)