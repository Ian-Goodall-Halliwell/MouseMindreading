# Import required libraries
import torch
from wrangler import preprocess
from wranglergoofy import preprocess as pp2
import optuna
from model import RNNmodel
from sklearn.preprocessing import LabelEncoder

# Set the device to CUDA (GPU) if available; otherwise, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the random seed for reproducibility
torch.manual_seed(47)

# Define the data version and region (for preprocessing)
ver = 'regions'
region = 'MOs'

# Load the data based on the specified version and region
if ver == 'regions':
    testrundata = preprocess(10, ver=ver, reg=region)
    feats = testrundata['features']
    labels = testrundata['labels']
    mins = testrundata['cutoffs']
    X = feats
    y = labels

    # Subsample the data to a fixed size for each region (optional)
    # for en, sample in enumerate(X):
    #     indexsample = np.random.choice(range(sample.shape[0]), size=mins[region])
    #     X[en] = sample[indexsample]

elif ver == 'single-session':
    testrundata = preprocess(10, ver=ver)
    feats = testrundata['features']
    labels = testrundata['labels']
    X = feats['Cori2016-12-14']
    y = labels['Cori2016-12-14']

elif ver == 'goofy':
    data = pp2()
    X = data['features']
    y = data['labels']

# Encode the target labels using LabelEncoder to convert them into numerical values
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

def optim(trial):
    """
    Function to optimize hyperparameters using Optuna.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial.

    Returns:
        float: Mean F1 score across all cross-validation folds for the given hyperparameters.
    """
    # Set the hyperparameters using Optuna suggestions
    cutv = trial.suggest_categorical('cut', [10, 20, 30, 40, 50, 60, 80, 100])
    X_t = [v[:, -cutv:] if v.shape[1] > cutv else v for v in X]
    input_size = 75  # Dimensionality of the input features
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256, 512])  # Number of units in the hidden layer
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4, 5, 6])  # Number of recurrent layers
    model_arch = trial.suggest_categorical('model_arch', ['lstm', 'gru', 'rnn'])
    num_epochs = 2000
    batch_size = 250
    dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2])
    l2 = trial.suggest_categorical('l2', [0, 0.0001, 0.001, 0.01, 0.1, 1])
    lr = trial.suggest_categorical('lr', [0.00001, 0.0001, 0.001, 0.01])
    output_size = 1  # Number of classes (binary)
    
    # Create the RNNmodel instance
    model = RNNmodel(input_size, hidden_size, num_layers, output_size, model_arch, num_epochs, batch_size, dropout, l2, lr, device)
    
    # Perform cross-validation and return the mean F1 score
    return model.cross_validate(X_t, y)

# Create or load the Optuna study for hyperparameter optimization
study_name = "flatbrain"  # Unique identifier of the study
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name, direction="maximize", study_name=study_name, load_if_exists=True)

# Optimize the hyperparameters using Optuna
study.optimize(optim)