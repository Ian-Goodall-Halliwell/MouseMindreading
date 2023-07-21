# plotting functions

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

max_t = 2500
cmap = matplotlib.cm.get_cmap('cividis')
col_yellow = cmap(0.9)
color = [[.9, 0, 0]]


def raster(spiketimes, i_trial=None, i_neuron=None, ax=None):
    """Spike raster plot of a certain neuron (all trials) or a certain trial (all neurons) given spiketimes of a
    specific session (i.e. dat_st[i_session]['ss'])

    :param spiketimes: spike times of a specific session with the dimensions (n_neurons, n_trials),
        e.g. pass dat_st[0]['ss'] for first session
    :param i_trial: index of trial for which all neurons should be plotted. leave None to plot all trials of i_neuron
    :param i_neuron: index of neuron for which all trials should be plotted. leave None to plot all neurons of i_trial
    :param ax: axes object to plot in. If None, a new figure will be created. If passing both i_trial and i_neuron ax
        must be a list of two elements, as two subßlots will be created.
    :return: ax: axes object (or list of two axes objects)
    """

    # check inputs
    if i_trial is not None and i_neuron is not None:
        assert not ax or len(ax) == 2
    assert i_trial or i_neuron, "must pass an index to a specific trial (i_trial) and/or neuron (i_neuron)"

    # spike raster plot for one trial (all neurons) and one neuron (all trials)
    if not ax:
        if i_trial is not None and i_neuron is not None:
            fig, ax = plt.subplots(2, 1, layout='constrained')
        else:
            fig, ax = plt.subplots(1, 1, layout='constrained')
            ax = [ax, ax]  # duplicate axes object into list so we can use ax[0] and ax[1] below regardless of # of axes
    else:
        ax = [ax, ax]
    dot_size = 3

    # plot spikes from all neurons for a specific trial
    if i_trial is not None:
        for i in range(spiketimes.shape[0]):
            col = color[0] if i == i_neuron else 'k'
            ax[0].plot(spiketimes[i][i_trial], np.ones(len(spiketimes[i][i_trial]))*i, '.', color=col,
                       markersize=dot_size)
        ax[0].set_title('Trial #' + str(i_trial))
        ax[0].set_xlim((-.01*max_t, 1.01*max_t))
        ax[0].set_ylim((-.02*spiketimes.shape[0], 1.02*spiketimes.shape[0]))
        ax[0].set_xlabel('Time [ms]')
        ax[0].set_ylabel('Neuron index')
    # plot spikes from all trials for a specific neuron
    if i_neuron is not None:
        for i, trl in enumerate(spiketimes[i_neuron]):
            col = color[0] if i == i_trial else 'k'
            ax[1].plot(trl, np.ones(len(trl))*i, '.', color=col, markersize=dot_size)
        ax[1].set_title('Neuron #' + str(i_neuron))
        ax[1].set_xlim((-.01*max_t, 1.01*max_t))
        ax[1].set_ylim((-.02*spiketimes.shape[1], 1.02*spiketimes.shape[1]))
        ax[1].set_xlabel('Time [ms]')
        ax[1].set_ylabel('Trial index')

    return ax
