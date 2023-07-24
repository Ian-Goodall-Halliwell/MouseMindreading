import numpy as np
import matplotlib.pyplot as plt
import plots
import util

# load spike data (run download_lfp.py first)
# todo: uncomment the following line (and comment the one below) to load just the first session to speed up testing
# dat_st = np.load("dat_test_s0.npy", allow_pickle=True)
dat_st = np.load("dat_st.npy", allow_pickle=True)
i_session = 0
spiketimes = dat_st[i_session]['ss']
spiketimes = spiketimes * 1000  # seconds -> milliseconds

# bin spikes
bin_size = 10
max_t = 2500
# spikebins, edges = util.bin_spiketimes(spiketimes, bin_size=bin_size, max_t=max_t, offset_step=2)
spikebins, edges = util.bin_spiketimes(spiketimes, bin_size=bin_size, max_t=max_t, offset_step=int(bin_size/5))

# just plots below here ###########################################################
# spike raster plot for one trial (all neurons) and one neuron (all trials)
i_trial = 0
i_neuron = 650
ax = plots.raster(spiketimes, i_trial=i_trial, i_neuron=i_neuron)

# plot all offsets for one neuron & trial
ax = plots.rebin_offsets(spikebins, edges, i_trial=i_trial, i_neuron=i_neuron)

plt.show()
