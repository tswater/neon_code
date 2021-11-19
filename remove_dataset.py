# Removes a Dataset from the Hdf5 files
import h5py
import os
files=['ABBY_L2.h5'] #blank implies all
dataset='SWC_005'
if len(files)==0:
    files=os.listdir('../L2')
for file in files:
    print(file,end='')
    with h5py.File('../L2/'+file,"a") as f:
        try:
            del f[dataset]
        except:
            print('fail',end='')
            continue
    print()
