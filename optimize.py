# Import required libraries
import torch
from wrangler_new import preprocess
from wranglergoofy import preprocess as pp2
from wrangler import preprocess as pp1
import optuna
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from model import RNNmodel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle as pkl
import os
from matplotlib import pyplot as plt
from sklearn import metrics
from torchsummary import summary

# Set the device to CUDA (GPU) if available; otherwise, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def scale(x):
    X_std = (x - np.min(x)) / (np.max(x) - (np.min(x) + 1e-5))
    X_scaled = X_std * (10 - 0) + 0
    return X_scaled
# Set the random seed for reproducibility


# Define the data version and region (for preprocessing)
ver = 'regions'
region = 'MOs'

# Load the data based on the specified version and region
if ver == 'regions':
    if not os.path.exists('datacheckpoint.pkl'):
        
        testrundata = preprocess(50, ver=ver, reg=region,thresh=0.25)
        #testrundata = pp1(10, ver=ver, reg=region)
        feats = testrundata['features']
        labels = testrundata['labels']
        mins = testrundata['cutoffs']
        X = feats
        Y = labels
        
        with open('datacheckpoint.pkl','wb') as f:
            pkl.dump((X,Y),f)
    else:
        with open('datacheckpoint.pkl','rb') as f:
            X,Y = pkl.load(f)
    
    # uniq = np.unique()
    # # Subsample the data to a fixed size for each region (optional)
    # for en, sample in enumerate(X):
    #     indexsample = np.random.choice(range(sample.shape[0]), size=mins[region])
    #     X[en] = sample[indexsample]

elif ver == 'single-session':
    testrundata = preprocess(10, ver=ver)
    feats = testrundata['features']
    labels = testrundata['labels']
    X = feats['Cori2016-12-14']
    y = labels['Cori2016-12-14']
    ham_downsample = resample(y,
             replace=True,
             n_samples=len(y),
             random_state=42)
elif ver == 'goofy':
    data = pp2()
    X = data['features']
    Y = data['labels']

# Encode the target labels using LabelEncoder to convert them into numerical values
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
print(np.mean(Y))
def optim(trial):
    """
    Function to optimize hyperparameters using Optuna.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial.

    Returns:
        float: Mean F1 score across all cross-validation folds for the given hyperparameters.
    """
    torch.manual_seed(47)
    # Set the hyperparameters using Optuna suggestions
    cutv = trial.suggest_categorical('cut', [10, 20, 30, 40, 50, 60, 80, 100])
    X_t = [v[:, -cutv:] if v.shape[1] > cutv else v for v in X]
    Y = globals()['Y']
    zz = [(scale(v),vv) for v,vv in zip(X_t,Y) if np.max(v) > 0]
    X_t = [v[0] for v  in zz]
    Y = [v[1] for v in zz]
    rus = RandomUnderSampler(random_state=42)
    x_ = np.array(list(range(len(X_t))))
    x_, Y = rus.fit_resample(x_.reshape(-1, 1), Y)
    X_t = [X_t[i[0]] for i in x_]
    input_size = 15  # Dimensionality of the input features
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256, 512,1024])  # Number of units in the hidden layer
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4, 5, 6])  # Number of recurrent layers
    model_arch = 'lstm' #trial.suggest_categorical('model_arch', ['lstm', 'gru', 'rnn'])
    num_epochs = 2000
    batch_size = 500
    dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2])
    l2 = 0 # trial.suggest_categorical('l2', [0, 0.0001, 0.001, 0.01, 0.1, 1])
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01, 0.1])
    output_size = 1  # Number of classes (binary)
    
    # Create the RNNmodel instance
    model = RNNmodel(input_size, hidden_size, num_layers, output_size, model_arch, num_epochs, batch_size, dropout, l2, lr, device)
    # zz = [(scale(v),vv) for v,vv in zip(X_t,Y) if np.max(v) > 0]
    # X_t = [v[0] for v  in zz]
    # y = [v[1] for v in zz]
    y = Y
    # Perform cross-validation and return the mean F1 score
    score, allpred,allreal = model.cross_validate(X_t, y)
    allpred, allreal = np.concatenate(allpred), np.concatenate(allreal)
    fpr, tpr, _ = metrics.roc_curve(allreal,  allpred)
    met = metrics.roc_auc_score(allreal,allpred)
    
    plt.close()
    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'figs/auroc_{met}_f1_{score}.png')
    plt.close()
    plt.close()
    cfig = metrics.ConfusionMatrixDisplay.from_predictions(allreal,np.rint(allpred))
    cfig.plot(cmap='viridis')  # You can change the colormap as desired
    plt.title('Confusion Matrix')
    plt.savefig(f'figs/confmatrix_f1_{score}.png')
    plt.close()
    
    #summary(model.model,(input_size,cutv))
    #plt.savefig(f'figs/networkarch_{score}.png')
    #plt.close()
    return score

# Create or load the Optuna study for hyperparameter optimization
study_name = "flatbrain"  # Unique identifier of the study
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name, direction="maximize", study_name=study_name, load_if_exists=True)

# Optimize the hyperparameters using Optuna
#besttrial = study.best_params
#study.enqueue_trial(besttrial)
study.optimize(optim)