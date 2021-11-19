# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_CLIM.py ------------------- #
# ---------------------------------------------------------------- #
# add extra tower climate info etc.
# name: 1
# domain: 0
# mean temp: 23
# mean precip: 24
# green dyas : 38

import numpy as np
import h5py
import datetime
import os
import csv

from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
neon_file   = '/home/tsw35/soteria/data/NEON/chm/'
base_dir   = '../../L2/'

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
for site in sites:
	if len(site)>4:
		continue
	else:
		


#fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
#fp_out.attrs['domain']=row[0]
#fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())



	
	
		
