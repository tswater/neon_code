# Removes a Dataset from the Hdf5 files
import h5py
import os
files=[] #blank implies all
datasets=['DUDZ','DTDZ']
base_dir   = '/home/tsw35/soteria/data/eddy_v2/lst/'
if len(files)==0:
    files=os.listdir(base_dir)
for dataset in datasets:
    for file in files:
        print(file,end='')
        with h5py.File(base_dir+file,"a") as f:
            try:
                del f[dataset]
            except:
                print('fail',end='')
                continue
        print()
