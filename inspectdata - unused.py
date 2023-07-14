import tarfile
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import shutil
import tqdm

def extract_and_save(x,filepath):
    
    tf = tarfile.open(os.path.join("data",x),"r:")
    x = x.rsplit(".",1)[0]
    tf.extractall(os.path.join("extracted_data",x))
    tf.close()
    for file in os.listdir(os.path.join("extracted_data",x)):
        filext = file.rsplit(".",1)[-1]
        if filext == "npy":
            try:
                dat = pd.DataFrame(np.load(os.path.join("extracted_data",x,file)))
                dat.to_csv(os.path.join("extracted_data",x,file.rsplit(".",1)[0] + ".csv"))
            except:
                dat = np.load(os.path.join("extracted_data",x,file))
                np.save(os.path.join("extracted_data",x,file.rsplit(".",1)[0] + str(dat.shape)), dat)
        elif filext == "tsv":
            dat= pd.read_csv(os.path.join("extracted_data",x,file),sep='\t')
            dat.to_csv(os.path.join("extracted_data",x,file.rsplit(".",1)[0] + ".csv"))
        os.remove(os.path.join("extracted_data",x,file))
        
path = "allData.tar" 
if os.path.exists("extracted_data"):
    shutil.rmtree("extracted_data")
    os.mkdir("extracted_data")
else:
    os.mkdir("extracted_data")
    
tf = tarfile.open(path,"r:")
tf.extractall("data")
tf.close()
Parallel(n_jobs=8,backend="loky")(
    delayed(extract_and_save)(
        x,
        path
        ) for x in tqdm.tqdm(os.listdir("data"))
    ) 
shutil.rmtree("data")
print("All done")