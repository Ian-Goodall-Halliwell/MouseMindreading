import numpy as np
import matplotlib.pyplot as plt

# load spike data (run download_lfp.py first)
dat_st = np.load("dat_st.npy", allow_pickle=True)

# test: plot spikes from first session, first trial for all neurons
i_nrn = 102
i_trl = 0
fig, ax = plt.subplots(2, 1)
for i in range(len(dat_st[0]['ss'][:][i_trl])):
    col = '.r' if i == i_nrn else '.k'
    ax[0].plot(dat_st[0]['ss'][i][i_trl], np.ones(len(dat_st[0]['ss'][i][i_trl]))*i, col)
ax[0].set_ylabel('neuron index')
# test: plot spikes from first session, all trials for a specific neuron
for i, trl in enumerate(dat_st[0]['ss'][i_nrn]):
    col = '.r' if i == i_trl else '.k'
    ax[1].plot(trl, np.ones(len(trl))*i, col)
ax[1].set_title('neuron #' + str(i_nrn))
ax[1].set_ylabel('trial index')
[a.set_xlabel('time [s]') for a in ax]
