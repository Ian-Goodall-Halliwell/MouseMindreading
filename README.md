# NMAproject

Neuronal Mouse Behavior Detection
=================================

Welcome to the repository for detecting mouse behavior based on neuronal spiking data from the Steinmetz dataset. This project includes Python scripts for preprocessing data, optimizing hyperparameters, and implementing a recurrent neural network (RNN) model. The repository is structured as follows:

Files
-----

1.  optimize.py

    -   Script for optimizing hyperparameters using Optuna.
    -   Imports custom preprocessing script (`wrangler.py`) and a custom RNN model (`model.py`).
    -   Utilizes GPU (if available) and sets random seeds for reproducibility.
    -   Defines a function to scale input features.
    -   Loads or preprocesses data based on the specified version and region.
    -   Encodes target labels using LabelEncoder.
    -   Implements a function (`optim`) for hyperparameter optimization using Optuna.
    -   Saves optimization results in the 'flatbrain.db' database.
2.  wrangler.py

    -   Custom preprocessing script.
    -   Downloads required data using `dl_alldata()` and `dl_st()` functions.
    -   Implements preprocessing steps, including binning spike times and filtering based on wheel movement.
    -   Defines a function (`preprocess`) for preprocessing neuronal spiking data.
3.  util.py

    -   Utility functions used in the preprocessing pipeline.
    -   Includes functions for binning spike times and applying transformations to data based on provided coordinates.
4.  model.py

    -   Defines the RNN model (`RNNClassifier`) and the RNN model wrapper (`RNNmodel`).
    -   Implements training, prediction, and cross-validation methods.
    -   Utilizes PyTorch for model implementation.
    -   Custom collate function for DataLoader is defined to handle variable-length sequences.
5.  download.py

    -   Script for downloading the Steinmetz dataset.

Getting Started
---------------

1.  Clone the repository: `git clone https://github.com/yourusername/mouse-behavior-detection.git`
2.  Install the required dependencies: `pip install -r requirements.txt`
3.  Run the optimization script: `python optimize.py`

Notes
-----

-   Make sure to have GPU support for faster training; otherwise, the code will default to CPU.
-   Adjust the parameters and configurations in the scripts based on your specific use case.

File Tree
-----

`download.py` - downloads steinmetz data files and concats to alldata.py

`download_lfp.py` - downloads some slightly preprocessed steinmetz data

`wrangler.py` - wrangles all the steinmetz data into a usable numpy array

`model.py` - contains all model code for the RNN

`util.py` - contains some utility functions for rebinning

`plots.py` - contains plotting funcitons for rebinning and raster plots

`rebin.py` - contains rebinning functions, used for data augmentation

`movement_onset_detection.py` - contains a function to perform movement initiation detection, used to determine start time of relevant data

`flatbrain.db` - contains optimization history

`datacheckpoint.pkl` - a saved version of the wrangler output, used to save time but can be deleted to regenerate data

`environment.yml` - conda environment needed to replicate results

`figs` - figure output folder

`tutorials` - some ipynb files used for inspiration
