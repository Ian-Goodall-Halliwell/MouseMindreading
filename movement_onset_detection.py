# -*- coding: utf-8 -*-
"""Movement Onset Detection Algorithm

Created by John Buggeln, July 07, 2023.
For use with the Steinmetz, 2019 dataset.
"""

import math as math
import numpy as np

# For progress bars
from tqdm.notebook import tqdm

#-- Calculating movement onset time --#

def movement_onset(mouse_vels, main_time = 100, stds = 2, time_buffer = 12):
  """Takes in an array of velocity data, and returns the index of movement onset,
  using a dual threshold method. Movement onset is calculated by taking the noise
  baseline as first 500ms of data before stimulus presentation (in the Steinmetz data set),
  and then searching the data for the absolute value of the first velocity that exceeds
  the mean absolute value baseline by n 'stds' (first threshold).
  In order for this to be considered a valid onset, the velocity above baseline
  must be maintained for 'main_time' (our second threshold.).

  Args:
    mouse_vels, dictionary of wheel velocity data for every mouse. Items in the dictionary
      are the wheel velocities binned every 10ms.
    stds, float, the number of standard deviations above baseline for the first threshold.
    main_time, float, the time window in ms of the second threshold.
    time_buffer, int, the number of data points after visual stimulus we exclude from
      our onset detection. We exclude data because we don't want to include false,
      or reflexive movements.
  Returns:
    onset_indices, dictionary, will return the movement onset indicies for each mouse,
      as a 1D array with the same dimensionality as the number of trials. The indices correspond to
      the 10 ms bins that the mouse velocity data contains.
  """

  # Initializingg our dual threshold method
  print("Number of std's above baseline for first threshold: ", stds)
  datapoints_to_check = int(main_time/10) # divide by 10, because timebins are 10ms
  print('Number of data points for second threshold: ', datapoints_to_check)

  # Selecting a time window buffer.
  time_buffer = 10 # number of data points (points * 10 = ms)

  # Constructing our dictionary of mice from the keys in the incoming dictionary of mice
  onset_indices = {mouse_key: [] for mouse_key in mouse_vels.keys()}

  # Calculating our baseline movement information for the all mice:
  all_mouse_data = np.concatenate([mouse_vels[mouse] for mouse in mouse_vels.keys()], axis = 0)
  trial_baseline_means = np.mean(np.abs(all_mouse_data[:, :50]), axis = 0) # taking abs to rectify our movement signals
  baseline_avg_abs = np.abs(np.mean(trial_baseline_means))
  baseline_std_avg = np.abs(np.std(trial_baseline_means))

  for mouse in tqdm(mouse_vels.keys()):
    cur_mouse = mouse_vels[mouse]

    # Iterating through every trial for each mouse
    for trial_idx in range(cur_mouse.shape[0]):

      # Setting our pointers
      cur_trial = cur_mouse[trial_idx, :]
      stored_idx = 0 # re/setting a pointer for a potential onset index


      # Slicing out our data post visual stimulus
      post_vis_stimulus =  cur_trial[50+time_buffer:-1]


      #-- Implementing our dual-threshold algorithm --#
      found_index = False # found_index flag
      for idx, vel in enumerate(post_vis_stimulus):
        if np.abs(vel) > (baseline_avg_abs + stds*baseline_std_avg):
          stored_vel = post_vis_stimulus[idx]
          stored_idx = idx
          num_trials_maintained = 0 # initializing our counter

          # Checking to see if our threshold has been maintained long enough
          for vel in post_vis_stimulus[stored_idx:stored_idx + datapoints_to_check]:
              if stored_vel > 0:
                if vel > (baseline_avg_abs + stds*baseline_std_avg):
                  num_trials_maintained += 1
              elif stored_vel < 0:
                if vel < -(baseline_avg_abs + stds*baseline_std_avg):
                  num_trials_maintained += 1


              # If maintained, we store the index for that trial
              if num_trials_maintained == datapoints_to_check:
                onset_indices[mouse].append(stored_idx + 50 + time_buffer) # we adde 50 to compensate for the baseline period that we removed
                found_index = True
                break

        if found_index == True:
          break
        elif stored_idx + datapoints_to_check > len(post_vis_stimulus):
          break

      # appending a NaN to the onset indices to track a failure to detect
      if found_index == False:
        onset_indices[mouse].append(float("NaN"))


  # Calculating the number of 'NaNs' (failures to detect movement onset)
  num_of_nans = 0
  for key in onset_indices:
    num_of_nans += sum(math.isnan(x) for x in onset_indices[key])
  print("Total Failures to detect onset: ", num_of_nans)

  return onset_indices
