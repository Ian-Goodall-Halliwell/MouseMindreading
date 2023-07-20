import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from movement_onset_detection import movement_onset
def preprocess():
    """
    Preprocesses the data.

    Returns:
        dict: Dictionary containing the preprocessed features, labels, and cutoffs.
    """
    alldat = np.load("alldata.npy", allow_pickle=True)

    allgoques = []
    for x in alldat:
        goquetimes = x['gocue'].copy().squeeze()
        goquetimes *= 100
        goquetimes = goquetimes.astype(int)
        goquetimes += 50
        allgoques.append(min(goquetimes))

    mintime = min(allgoques)
    print(f"Maximum data length = {mintime}")

    outputdata = []
    outputlabel = []
    mov = movement_onset({x['mouse_name'] + x['date_exp']:x['wheel'].squeeze() for x in alldat})
    for x in alldat:
        spikes = np.transpose(x["spks"], axes=(1, 0, 2))
        wheel = x["wheel"].squeeze()
        # goquetimes = x['gocue'].copy().squeeze()
        # goquetimes *= 100
        # goquetimes += 50
        # goquetimes = goquetimes.astype(int)
        starttime = 50
        movementset = mov[x['mouse_name'] + x['date_exp']]
        
        # resptimes = x['response_time'].copy().squeeze()
        # resptimes *= 100
        # resptimes += 50
        # resptimes = resptimes.astype(int)

        def _f(goquetimes, resptimes, wheel, xv, mintime, x, e):
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

            maxpregodata = xv[:, goquetimes:resptime]
            return goofit(maxpregodata, x['ccf']), wheeldir

        with Parallel(n_jobs=12, backend='threading') as parallel:
            data = parallel(delayed(_f)(
                starttime,
                movementset,
                wheel,
                xv,
                mintime,
                x,
                e,
            ) for e, xv in enumerate(spikes))

        [outputdata.append(x[0]) for x in data if x is not None]
        [outputlabel.append(x[1]) for x in data if x is not None]

    return {"features": outputdata, "labels": outputlabel, "cutoffs": None}


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
        scaler = MinMaxScaler((0, 49))
        return scaler.fit_transform(x.reshape(-1, 1))

    coords = np.apply_along_axis(shrink, 0, coords).squeeze()
    coords = (coords).astype(int)
    coords = np.where(coords < 0, 0, coords)
    allouts = np.zeros((3, 50, x.shape[1]))

    for ax in range(coords.shape[1]):
        axcoord = coords[:, ax]
        output = np.zeros((50, x.shape[1]))

        for e, v in enumerate(axcoord):
            output[v] += x[e]

        allouts[ax] = output

    return allouts.reshape(-1, allouts.shape[-1])


if __name__ == "__main__":
    preprocess(ver='goofy')