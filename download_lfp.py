import numpy as np
import os, requests

def dl_st():
  fname = ['steinmetz_st.npz']
  # fname.append('steinmetz_wav.npz')
  # fname.append('steinmetz_lfp.npz')

  url = ["https://osf.io/4bjns/download"]
  # url.append("https://osf.io/ugm9v/download")
  # url.append("https://osf.io/kx3v9/download")

  for j in range(len(url)):
    if not os.path.isfile(fname[j]):
      try:
        r = requests.get(url[j])
      except requests.ConnectionError:
        print("!!! Failed to download data !!!")
      else:
        if r.status_code != requests.codes.ok:
          print("!!! Failed to download data !!!")
        else:
          with open(fname[j], "wb") as fid:
            fid.write(r.content)
  # dat_LFP = np.load('steinmetz_lfp.npz', allow_pickle=True)['dat']
  # dat_WAV = np.load('steinmetz_wav.npz', allow_pickle=True)['dat']
  dat_ST = np.load('steinmetz_st.npz', allow_pickle=True)['dat']

  for j in range(len(fname)):
    os.remove('steinmetz_st.npz')
  np.save("dat_st",dat_ST)
