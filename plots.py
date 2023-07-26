# plotting functions

import os
import glob
from PIL import Image
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


def rebin_offsets(spikebins, edges, i_trial=0, i_neuron=0, ax=None):
    """Plot the histograms of rebinned spiketimes for an example neuron and trial - one subplot for each offset.

    :param spikebins: list of numpy arrays, containing results of util.bin_spiketimes() for different offsets
    :param edges: list of arrays of bin edges used for binning - as returned by util.bin_spiketimes()
    :param i_trial: index of trial for which all neurons should be plotted.
    :param i_neuron: index of neuron for which all trials should be plotted.
    :param ax: list of axes objects to plot in (same length as spikebins). If None, a new figure will be created
    :return: ax: list of axes objects
    """

    assert ax is None or len(ax) == len(spikebins), "ax must be None or a list of same length as spikebins"

    # plot all offsets for one neuron & trial
    if not ax:
        fig, ax = plt.subplots(len(spikebins), 1, layout='constrained')
    for off in range(len(spikebins)):
        ax[off].stairs(spikebins[off][i_neuron][i_trial], edges[off])
        # ax[off].bar(edges[off][:-1] + bin_size/2, spikebins[off][i_neuron][i_trial], width=bin_size)
        ax[off].set_xlim((0, max_t))
        ax[off].set_ylabel('off: ' + str(edges[off][0]))
    [ax[o].set_xticklabels('') for o in range(len(spikebins) - 1)]
    ax[-1].set_xlabel('Time [ms]')

    return ax


def rebin_matrix(spikebins, edges, i_trial=None, i_neuron=None):
    """Plot the matrix plot of rebinned spiketimes of a certain neuron (all trials) or a certain trial (all neurons)

    :param spikebins: list of numpy arrays, containing results of util.bin_spiketimes() for different offsets
    :param edges: list of arrays of bin edges used for binning - as returned by util.bin_spiketimes()
    :param i_trial: index of trial for which all neurons should be plotted. leave None to plot all trials of i_neuron
    :param i_neuron: index of neuron for which all trials should be plotted. leave None to plot all neurons of i_trial
    :return: figs: list of figure objects
    """

    # check inputs
    assert i_trial is None or i_neuron is None, "either i_trial or i_neuron must be None"
    assert i_trial or i_neuron, "must pass an index to a specific trial (i_trial) and/or neuron (i_neuron)"
    i_example_hist = 100  # if i_trial==None, this will be the example trial for the histogram, otherwise the neuron

    # get absolute maximum of all bins for all offsets (both for
    max_spikebins = np.max([np.max(s) for s in spikebins])  # todo: max_spikebins is wrong
    if i_trial is None:
        max_hist = np.max([np.max(s[i_neuron][i_example_hist]) for s in spikebins])
    elif i_neuron is None:
        max_hist = np.max([np.max(s[i_example_hist][i_trial]) for s in spikebins])

    # plot all offsets for one neuron & trial
    figs = []
    for off in range(len(spikebins)):
        fig, ax = plt.subplots(2, 2, layout='constrained', height_ratios=(10, 1), width_ratios=(20, 1))

        if i_trial is None:
            mat_data = spikebins[off][i_neuron][:]
            mat_ylabel = 'Trial #'
            rebin_offsets([spikebins[off]], [edges[off]], i_trial=i_example_hist, i_neuron=i_neuron, ax=[ax[1][0]])
        elif i_neuron is None:
            mat_data = spikebins[off][:][i_trial]
            mat_ylabel = 'Neuron #'
            rebin_offsets([spikebins[off]], [edges[off]], i_trial=i_trial, i_neuron=i_example_hist, ax=[ax[1][0]])
        h_mat = ax[0][0].imshow(mat_data, aspect='auto', cmap='cividis')
        ax[0][0].set_ylabel(mat_ylabel)
        h_cbar = plt.colorbar(h_mat, cax=ax[0][1])
        # h_cbar.set_ticks(list(range(max_spikebins+1)))  # todo: max_spikebins is wrong
        ax[1][0].set_ylim((0, max_hist + 0.5))
        ax[1][0].set_yticks((0, max_hist))
        ax[1][0].set_ylabel('N spikes')
        ax[1][0].set_xlabel('Time [ms]')
        ax[1][1].axis('off')
        figs.append(fig)

    return figs


def save_gif(figs, filename='gif', scaling_factor=1, frame_duration_ms=500, n_loops=0):
    """String a list of figures into a gif and save.

    :param figs: list of matplotlib.Figure objects
    :param filename:
    :param scaling_factor: width in pixels of the resulting .gif will be the minimum width of all images times this
    :param frame_duration_ms: duration of each .gif frame in milliseconds
    :param n_loops: number of loops the .gif will go through. 0 means infinitely looping, -1 means no loops
    :return:
    """

    print('saving images', end='')
    image_names = []
    for i, fig in enumerate(figs):
        image_names.append(str(i).zfill(3) + '.png')
        fig.savefig(image_names[i])
        print('.', end='')
        plt.close(fig)
    # loop through list of image files, open and append to list of frames
    print('\nreading images', end='')
    frames = []
    img_widths = []
    for img in image_names:
        frames.append(Image.open(img))
        img_widths.append(frames[-1].size[0])
        print('.', end='')
        os.remove(img)

    # resize frames to match
    print('\nresizing frames', end='')
    gif_width = img_widths[0] * scaling_factor
    for i, frm in enumerate(frames):
        frames[i] = frm.resize((round(gif_width), round(frm.size[1] * gif_width / frm.size[0])),
                               Image.Resampling.LANCZOS)
        print('.', end='')

    # save frames (in reversed order) into an infinitely looping gif
    print('\nsaving gif')
    frames[-1].save(
        filename + '_' + str(frame_duration_ms) + '_' + str(n_loops) + '_' +
        str(scaling_factor) + '.gif', format='GIF', append_images=frames[-2::-1],
        save_all=True, duration=frame_duration_ms, disposal=2, optimize=True, loop=n_loops)
