# Imports
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from wrangler import preprocess
from wranglergoofy import preprocess as pp2
import optuna
from model import RNNmodel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader

import pickle as pkl
ver = 'goofy'


if ver == 'regions':
    testrundata = preprocess(10,ver=ver)
    feats = testrundata['features']
    labels = testrundata['labels']
    mins = testrundata['cutoffs']

    region='MOs'
    X = feats[region]
    y = labels[region]

    #subsampling
    for en,sample in enumerate(X):
        
        indexsample = np.random.choice(range(sample.shape[0]),size=mins[region])
        X[en] = sample[indexsample]

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
print(np.mean(y))

X_train, X_test, y_train,  y_test = train_test_split(X,y,test_size=.30)


X = torch.tensor(np.array(X_train),dtype=torch.float32)
y = torch.tensor(np.array(y_train),dtype=torch.float32).reshape(-1,1)
X_test = torch.tensor(np.array(X_test),dtype=torch.float32)
y_test = torch.tensor(np.array(y_test),dtype=torch.float32).reshape(-1,1)
test = True
if test:
    input_size = 39  # Dimensionality of the input features
    hidden_size = 64  # Number of units in the hidden layer
    num_layers = 1  # Number of recurrent layers
    model_arch = 'lstm'
    num_epochs = 100
    batch_size = 100
    dropout = 0
    l2 = 0
    lr = 0.001
    output_size = 1  # Number of classes (binary)
    
    model = RNNmodel(input_size,hidden_size,num_layers,output_size,model_arch,num_epochs,batch_size,dropout,l2,lr)
    dataloader = DataLoader(list(zip(X, y)), batch_size=model.batch_size, pin_memory=True, pin_memory_device='cuda', shuffle=False)
    dataloader_true = DataLoader(list(zip(X_test, y_test)), batch_size=model.batch_size, pin_memory=True, pin_memory_device='cuda', shuffle=False)
    model.train(dataloader, dataloader_true)
    preds, all_reals = model.predict(dataloader_true)
    score = f1_score(all_reals, preds)
    print(score)

def optim(trial):
    # Set the hyperparameters
    input_size = 39  # Dimensionality of the input features
    hidden_size = trial.suggest_categorical('hidden_size',[32,64,128,256])  # Number of units in the hidden layer
    num_layers = trial.suggest_categorical('num_layers',[1,2])  # Number of recurrent layers
    model_arch = 'lstm'
    num_epochs = num_epochs = 100
    batch_size = 100
    dropout = trial.suggest_categorical('dropout',[0,0.1])
    l2 = trial.suggest_categorical('l2',[0,0.0001,0.001,0.01])
    lr = 0.01
    output_size = 1  # Number of classes (binary)
    model = RNNmodel(input_size,hidden_size,num_layers,output_size,model_arch,num_epochs,batch_size,dropout,l2,lr)
    return model.cross_validate(X,y)
    
    
    
study_name = "flatbrain"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name,direction="maximize",study_name=study_name,load_if_exists=True)
study.optimize(optim)