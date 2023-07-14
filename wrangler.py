import numpy as np
def preprocess(minamnt=30):
    alldat = np.load("alldata.npy", allow_pickle=True)
    allgoques = []
    allneurs = {}
    for x in alldat:
        goquetimes = x['gocue'].copy().squeeze()
        goquetimes *= 100
        goquetimes = goquetimes.astype(int)
        goquetimes += 50
        #allneurs.append(x["spks"].shape[0])
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
    outputdata = {}
    outputlabel = {}
    for x in alldat:
        spikes = np.transpose(x["spks"],axes=(1,0,2))
        wheel = x["wheel"].squeeze()
        goquetimes = x['gocue'].copy().squeeze()
        goquetimes *= 100
        goquetimes += 50
        goquetimes = goquetimes.astype(int)
        regs = x['brain_area']
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
            regions = {}
            for reg in np.unique(regs):
                if reg == 'root':
                    continue
                regiondata = maxpregodata[regs == reg]
                try:
                    if regiondata.shape[0] < allneurmin[reg]:
                        continue
                except:
                    continue
                try:
                    outputdata[reg].append(regiondata)
                    outputlabel[reg].append(wheeldir)
                except:
                    outputdata[reg] = []
                    outputdata[reg].append(regiondata)
                    outputlabel[reg] = []
                    outputlabel[reg].append(wheeldir)

    # import pickle as pkl
    # with open('currdatas.pkl', 'wb') as f:
    #     pkl.dump({"features":outputdata,"labels":outputlabel,"cutoffs":allneurmin},f)
    return {"features":outputdata,"labels":outputlabel,"cutoffs":allneurmin}
#print('e')
    
    