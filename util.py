# utility functions
import numpy as np


def bin_spiketimes(spiketimes, bin_size=10, max_t=2500, offset=0):
    """Calculate binned spike counts from raw spike times of a specific session (i.e. dat_st[i_session]['ss'])

    :param spiketimes: spike times of a specific session with the dimensions (n_neurons, n_trials),
        e.g. pass dat_st[0]['ss'] for first session
    :param bin_size: width of bins in milliseconds
    :param max_t: end of binning (maximum time point)
    :param offset: start of the first bin in milliseconds
    :return:    hist: numpy array with same dimensions as spiketimes, containing results of np.histogram
                edges: array of bin edges used for binning - should be np.arange(offset, max_t, bin_size)
    """

    hist = np.zeros_like(spiketimes)
    bins = np.arange(offset, max_t, bin_size)

    for i, nrn in enumerate(spiketimes):
        for j, trl in enumerate(nrn):
            hist[i][j], edges = np.histogram(trl, bins=bins)

    return hist, edges
