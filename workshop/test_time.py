# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_DP04.py ------------------- #
# ---------------------------------------------------------------- #
# Add the important contents of the main product files to the base #
# h5 including stuff

import os
import netCDF4 as nc
import numpy as np
import h5py
import datetime
import ephem
import rasterio
import csv
import subprocess
from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------------- #
# USER INPUTS AND CONSTANTS #
# ------------------------- #
neon_dir = '/home/tsw35/soteria/data/NEON/dp04/'
base_dir = '/home/tsw35/soteria/data/eddy_v2/lst/'
other_dir= '/home/tsw35/tyche/data/GOES/pickles/1x1_arcdegree/'

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
sites=[sites[10]]
missed={2017:0,2018:0,2019:0,2020:0,2021:0,2022:0}
for site in sites[rank::size]:
    if len(site)>4:
          continue
    print(site+': ',end='',flush=True)
     
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    time=fp_out['TIME'][:]
    filelist=os.listdir(other_dir)
    for t in time:
        dt=datetime.datetime(1970,1,1,0,0)+datetime.timedelta(seconds=t)
        if dt<datetime.datetime(2017,5,25,0,0):
            continue
        if dt.minute==30:
            continue
        if ('cl_'+str(dt)+'p') in filelist:
            pass
        else:
            missed[dt.year]=missed[dt.year]+1
            print(dt)
print(missed)
