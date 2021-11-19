import h5py
import os
for file in os.listdir('../L2/'):
    fp=h5py.File('../L2/'+file)
    if 'SWCs' not in fp.keys():
        print(file)

