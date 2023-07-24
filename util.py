# utility functions
import numpy as np


def bin_spiketimes(spiketimes, bin_size=10, max_t=2500, offset_step=1):
    """Calculate binned spike counts from raw spike times of a specific session (i.e. dat_st[i_session]['ss'])

    :param spiketimes: spike times of a specific session with the dimensions (n_neurons, n_trials),
        e.g. pass dat_st[0]['ss'] for first session
    :param bin_size: width of bins in milliseconds
    :param max_t: end of binning (maximum time point)
    :param offset_step: timestep by which the offset (start of the first bin in milliseconds) will be increased
    :return:    hist: list of numpy arrays with same dimensions as spiketimes, containing results of np.histogram
                    for each offset between 0 and bin_size (incremented in steps of duration offset_step
                edges: list of arrays of bin edges used for binning - should be np.arange(offset, max_t, bin_size)
    """

    if offset_step is not None:
        offsets = np.arange(0, bin_size, offset_step)
    else:
        offsets = np.array([0])
    hist = []
    edge = []

    for o, offset in enumerate(offsets):
        hist.append(np.zeros_like(spiketimes))
        bins = np.arange(offset, max_t+bin_size, bin_size)

        for n, nrn in enumerate(spiketimes):
            for t, trl in enumerate(nrn):
                hist[o][n][t], edges = np.histogram(trl, bins=bins)
        edge.append(edges)

        # stack array so dimensions will be consistent with alldata
        hist[o] = np.array([np.stack(h) for h in hist[o]])

    return hist, edge
