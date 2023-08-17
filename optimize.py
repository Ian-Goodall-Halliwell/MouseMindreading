# Import required libraries
import torch
from wrangler import preprocess  # Importing a custom preprocessing script 
import optuna
from imblearn.under_sampling import RandomUnderSampler
from model import RNNmodel  # Importing a custom RNN model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle as pkl
import os
from matplotlib import pyplot as plt
from sklearn import metrics
import random

# Set the device to CUDA (GPU) if available; otherwise, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set a random seed for reproducibility
random_seed = 42
random.seed(random_seed)

# Set the random seed for numpy
np.random.seed(random_seed)

# Set the random seed for PyTorch's random number generator
torch.manual_seed(random_seed)

# Check if CUDA (GPU support) is available and set its random seed if applicable
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define a function named 'scale' to scale input features
def scale(x):
    X_std = (x - np.min(x)) / (np.max(x) - (np.min(x) + 1e-5))
    X_scaled = X_std * (10 - 0) + 0
    return X_scaled

# Define the data version and region for preprocessing
region = 'MOs'

# Load the data based on the specified version and region
if not os.path.exists('datacheckpoint.pkl'):
    testrundata = preprocess(50, reg=region, thresh=0.25)  # Preprocess data using 'preprocess' function
    feats = testrundata['features']  # Extract features from the preprocessed data
    labels = testrundata['labels']  # Extract labels from the preprocessed data
    mins = testrundata['cutoffs']    # Extract cutoffs from the preprocessed data
    X = feats  # Assign features to variable 'X'
    Y = labels  # Assign labels to variable 'Y'
    
    with open('datacheckpoint.pkl', 'wb') as f:
        pkl.dump((X, Y), f)  # Save features and labels to a binary file named 'datacheckpoint.pkl'
else:
    with open('datacheckpoint.pkl', 'rb') as f:
        X, Y = pkl.load(f)  # Load features and labels from the binary file 'datacheckpoint.pkl'

# Encode the target labels using LabelEncoder to convert them into numerical values
encoder = LabelEncoder()  # Create an instance of LabelEncoder
encoder.fit(Y)  # Fit the encoder on the labels
Y = encoder.transform(Y)  # Transform the labels to numerical values
print(np.mean(Y))  # Print the mean of the transformed labels

# Define a function named 'optim' to optimize hyperparameters using Optuna
def optim(trial):
    """
    Function to optimize hyperparameters using Optuna.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial.

    Returns:
        float: Mean F1 score across all cross-validation folds for the given hyperparameters.
    """

    # Set the hyperparameters using Optuna suggestions
    cutv = trial.suggest_categorical('cut', [10, 20, 30, 40, 50, 60, 80, 100])  # Suggest a value for 'cut' hyperparameter
    X_t = [v[:, -cutv:] if v.shape[1] > cutv else v for v in X]  # Truncate input features based on 'cut' value
    Y = globals()['Y']  # Access the global variable 'Y'
    zz = [(scale(v), vv) for v, vv in zip(X_t, Y) if np.max(v) > 0]  # Scale and filter data based on max value
    X_t = [v[0] for v in zz]  # Extract the scaled features
    Y = [v[1] for v in zz]  # Extract the labels
    rus = RandomUnderSampler(random_state=random_seed)  # Create an instance of RandomUnderSampler
    x_ = np.array(list(range(len(X_t))))  # Create an array for indices
    x_, Y = rus.fit_resample(x_.reshape(-1, 1), Y)  # Undersample the data
    X_t = [X_t[i[0]] for i in x_]  # Select the undersampled features
    
    # Define hyperparameters for the RNN model
    input_size = 15
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256, 512, 1024])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4, 5, 6])
    model_arch = 'lstm' 
    num_epochs = 2000
    batch_size = 500
    dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2])
    l2 = trial.suggest_categorical('l2', [0, 0.0001, 0.001, 0.01, 0.1, 1])
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01, 0.1])
    output_size = 1
    
    # Create an instance of the RNNmodel
    model = RNNmodel(input_size, hidden_size, num_layers, output_size, model_arch, num_epochs, batch_size, dropout, l2, lr, device)

    # Perform cross-validation and return the mean F1 score
    score, allpred, allreal = model.cross_validate(X_t, Y)
    allpred, allreal = np.concatenate(allpred), np.concatenate(allreal)
    fpr, tpr, _ = metrics.roc_curve(allreal, allpred)
    met = metrics.roc_auc_score(allreal, allpred)
    
    plt.close()
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'figs/auroc_{met}_f1_{score}.png')
    plt.close()
    plt.close()
    cfig = metrics.ConfusionMatrixDisplay.from_predictions(allreal, np.rint(allpred))
    cfig.plot(cmap='viridis')  # Create a confusion matrix plot
    plt.title('Confusion Matrix')
    plt.savefig(f'figs/confmatrix_f1_{score}.png')
    plt.close()

    return score

# Create or load the Optuna study for hyperparameter optimization
study_name = "flatbrain"  # Unique identifier of the study
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name, direction="maximize", study_name=study_name, load_if_exists=True)

# Optimize the hyperparameters using Optuna
besttrial = study.best_params
study.enqueue_trial(besttrial)
study.optimize(optim, n_trials=1)