import numpy as np
from movement_onset_detection import movement_onset
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from util import bin_spiketimes
from tqdm import tqdm
def preprocess(minamnt=10,ver="regions",reg=None,thresh=None):
    alldat = np.load("alldata.npy", allow_pickle=True)
    dat_st = np.load("dat_st.npy", allow_pickle=True)
    if ver == "regions":
        allgoques = []
        allneurs = {}
        for x in alldat:
            goquetimes = x['gocue'].copy().squeeze()
            goquetimes *= 100
            goquetimes = goquetimes.astype(int)
            goquetimes += 50

            regs = x['brain_area']
            for reg in np.unique(regs):
                regiondata = x["spks"][:,0][regs == reg]
                if regiondata.shape[0] < minamnt:
                    continue
                try:
                    allneurs[reg].append(regiondata.shape[0])
                except:
                    allneurs[reg] = []
                    allneurs[reg].append(regiondata.shape[0])
            allgoques.append(min(goquetimes))
        allneurmin = {x:min(allneurs[x]) for x in allneurs}
        mintime = min(allgoques)
        print(f"Maximum data length = {mintime}")
        outputdata = []
        outputlabel = []
        mov = movement_onset({x['mouse_name'] + x['date_exp']:x['wheel'].squeeze() for x in alldat})
        for endx,x in tqdm(enumerate(alldat)):
            # xtemp = [np.transpose(x["spks"],axes=(1,0,2))]
            #xtemp = [np.transpose(spikes_,axes=(1,0,2))]
            spikes = dat_st[endx]['ss']
            wheel = x["wheel"].squeeze()
            
            regs = x['brain_area']
            starttime = 50
            movementset = mov[x['mouse_name'] + x['date_exp']]
            
            spikes = spikes * 1000
            xtemp,_ = bin_spiketimes(spikes,bin_size=10,offset_step=2)
            xtemp = [np.transpose(vv,axes=(1,0,2)) for vv in xtemp]

            for z in xtemp:
                with Parallel(n_jobs=12, backend='loky') as parallel:
                    
                    data = parallel(delayed(_f)(
                        starttime,
                        movementset,
                        wheel,
                        xv,
                        mintime,
                        x,
                        e,
                        reg,
                        thresh,
                        minamnt,
                    ) for e, xv in enumerate(z))

                [outputdata.append(x[0]) for x in data if x is not None]
                [outputlabel.append(x[1]) for x in data if x is not None]
            
        
        return {"features":outputdata,"labels":outputlabel,"cutoffs":allneurmin}
    elif ver == "single-session":
        allgoques = []
        allneurs = {}
        for x in alldat:
            goquetimes = x['gocue'].copy().squeeze()
            goquetimes *= 100
            goquetimes = goquetimes.astype(int)
            goquetimes += 50

            
            allgoques.append(min(goquetimes))
        mintime = min(allgoques)
        print(f"Maximum data length = {mintime}")
        outputdata = {}
        outputlabel = {}
        for x in alldat:
            spikes = np.transpose(x["spks"],axes=(1,0,2))
            wheel = x["wheel"].squeeze()
            goquetimes = x['gocue'].copy().squeeze()
            goquetimes *= 100
            goquetimes += 50
            goquetimes = goquetimes.astype(int)
            resptimes = x['response_time'].copy().squeeze()
            resptimes *= 100
            resptimes += 50
            resptimes = resptimes.astype(int)
            for e,xv in enumerate(spikes):
                # POSITIVE MOVEMENT == LEFT
                # NEGATIVE MOVEMENT == RIGHT
                goque = goquetimes[e]
                resptime = resptimes[e]
                wheel_temp = wheel[e][goque:resptime]
                wheel_direction = np.mean(wheel_temp)
                if wheel_direction > 0:
                    wheeldir = 'left'
                elif wheel_direction < 0:
                    wheeldir = 'right'
                else:
                    wheeldir = 'nogo'
                    continue
                
                #pregodata = xv[:,:goque]
                maxpregodata = xv[:,50:mintime]
                
                
                try:
                    outputdata[x['mouse_name'] + x['date_exp']].append(maxpregodata)
                    outputlabel[x['mouse_name'] + x['date_exp']].append(wheeldir)
                except:
                    outputdata[x['mouse_name'] + x['date_exp']] = []
                    outputdata[x['mouse_name'] + x['date_exp']].append(maxpregodata)
                    outputlabel[x['mouse_name'] + x['date_exp']] = []
                    outputlabel[x['mouse_name'] + x['date_exp']].append(wheeldir)
    
    return {"features":outputdata,"labels":outputlabel,"cutoffs":None}
def _f(goquetimes, resptimes, wheel, xv, mintime, x, e,reg,thresh,minamnt):
                
    resptime = resptimes[e]
    if resptime != resptime:
        return
    wheel_temp = wheel[e][goquetimes:resptime]
    wheel_direction = np.mean(wheel_temp)
    if wheel_direction > 0:
        wheeldir = 'left'
    elif wheel_direction < 0:
        wheeldir = 'right'
    else:
        wheeldir = 'nogo'
        return
    regidxs = x['brain_area']
    regidxs = np.where(regidxs == reg, True,False)
    if np.sum(regidxs) < minamnt:
        return
    maxpregodata = xv[:, goquetimes:resptime]
    maxpregodata = maxpregodata[regidxs]
    ccfdata = x['ccf'][regidxs]
    contrast = abs(x['contrast_left'][e] - x['contrast_right'][e])
    
    if contrast >= thresh and np.sum(ccfdata) > 1:
        return goofit(maxpregodata, ccfdata), wheeldir
    else:
        pass
def goofit(x, coords):
    """
    Applies a transformation to the given data based on the provided coordinates.

    Args:
        x (numpy.ndarray): Input data.
        coords (numpy.ndarray): Coordinates for transformation.

    Returns:
        numpy.ndarray: Transformed data.
    """
    def shrink(x):
        scaler = MinMaxScaler((0, 4))
        return scaler.fit_transform(x.reshape(-1, 1))

    coords = np.apply_along_axis(shrink, 0, coords).squeeze()
    coords = (coords).astype(int)
    coords = np.where(coords < 0, 0, coords)
    allouts = np.zeros((3, 5, x.shape[1]))

    for ax in range(coords.shape[1]):
        axcoord = coords[:, ax]
        output = np.zeros((5, x.shape[1]))

        for e, v in enumerate(axcoord):
            output[v] += x[e]

        allouts[ax] = output

    return allouts.reshape(-1, allouts.shape[-1])
    