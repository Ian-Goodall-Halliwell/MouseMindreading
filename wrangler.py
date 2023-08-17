import numpy as np
from movement_onset_detection import movement_onset
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
def preprocess(minamnt=10,ver="regions",reg=None):
    alldat = np.load("alldata.npy", allow_pickle=True)
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
        for x in alldat:
            spikes = np.transpose(x["spks"],axes=(1,0,2))
            wheel = x["wheel"].squeeze()
            
            regs = x['brain_area']
            starttime = 50
            movementset = mov[x['mouse_name'] + x['date_exp']]
        def _f(goquetimes, resptimes, wheel, xv, mintime, x, e,reg):
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
            maxpregodata = xv[:, goquetimes:resptime]
            maxpregodata = maxpregodata[regidxs]
            ccfdata = x['ccf'][regidxs]
            return goofit(maxpregodata, ccfdata), wheeldir

        with Parallel(n_jobs=12, backend='threading') as parallel:
            data = parallel(delayed(_f)(
                starttime,
                movementset,
                wheel,
                xv,
                mintime,
                x,
                e,
                reg,
            ) for e, xv in enumerate(spikes))

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
        scaler = MinMaxScaler((0, 9))
        return scaler.fit_transform(x.reshape(-1, 1))

    coords = np.apply_along_axis(shrink, 0, coords).squeeze()
    coords = (coords).astype(int)
    coords = np.where(coords < 0, 0, coords)
    allouts = np.zeros((3, 10, x.shape[1]))

    for ax in range(coords.shape[1]):
        axcoord = coords[:, ax]
        output = np.zeros((10, x.shape[1]))

        for e, v in enumerate(axcoord):
            output[v] += x[e]

        allouts[ax] = output

    return allouts.reshape(-1, allouts.shape[-1])