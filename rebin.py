import numpy as np
import matplotlib.pyplot as plt
import plots
import util

# load spike data (run download_lfp.py first)
#dat_st = np.load("dat_st.npy", allow_pickle=True)
# todo: just loading first session to speed up testing
dat_st = np.load("dat_test_s0.npy", allow_pickle=True)
i_session = 0
spiketimes = dat_st[i_session]['ss']
spiketimes = spiketimes * 1000

# spike raster plot for one trial (all neurons) and one neuron (all trials)
i_trial = 0
i_neuron = 275
ax = plots.raster(spiketimes, i_trial=i_trial, i_neuron=i_neuron)

# bin spikes
spikebins, _ = util.bin_spiketimes(spiketimes, bin_size=50, max_t=2500, offset=0)
fig, ax = plt.subplots()
for trl in spikebins[i_neuron]:
    ax.plot(trl)

plt.show()
