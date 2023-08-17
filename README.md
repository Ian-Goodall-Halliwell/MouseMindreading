# NMAproject

download.py - downloads steinmetz data files and concats to alldata.py

download_lfp.py - downloads some slightly preprocessed steinmetz data

wrangler.py - wrangles all the steinmetz data into a usable numpy array

model.py - contains all model code for the RNN

util.py - contains some utility functions for rebinning

plots.py - contains plotting funcitons for rebinning and raster plots

rebin.py - contains rebinning functions, used for data augmentation

movement_onset_detection.py - contains a function to perform movement initiation detection, used to determine start time of relevant data

flatbrain.db - contains optimization history

datacheckpoint.pkl - a saved version of the wrangler output, used to save time but can be deleted to regenerate data

environment.yml - conda environment needed to replicate results

figs - figure output folder

tutorials - some ipynb files used for inspiration