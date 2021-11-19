# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_LAI.py ------------------- #
# ---------------------------------------------------------------- #
# Code Description for add_ndvi.py

import numpy as np
import h5py
import datetime
import os
import csv
import rasterio

from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
ndvi_dir  = '/home/tsw35/soteria/data/sentinel/raw_NDVI/'
base_dir  = '../../L2/'
neon_dir  = '/home/tsw35/soteria/data/NEON/raw_data/'

# -------------------- #
# HELPER FUNCTION SITE #
# -------------------- #
def checknearest(arr,x,y):
	zs = []
	for i in [-1,0,1]:
		for j in [-1,0,1]:
			z = arr[x+i,y+j]
			if z<100:
				zs.append(z)
	if len(zs)>0:
		return np.sum(zs)/(len(zs)+.01)
	else:
		return -9999


# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['ABBY']
for site in sites[rank::size]:
	if len(site)>4:
		continue
	if site == 'zips':
		continue
	print(site+': ',end='',flush=True)
	
	# Load in the base file
	fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
	time=fp_out['TIME'][:]
	lat=float(fp_out.attrs['lat'])
	lon=float(fp_out.attrs['lon'])
	
	# Load in the VCF
	try:
		fp_ndvi = h5py.File(ndvi_dir+site+'_ndvi.h5','r')
	except Exception:
		ssd_ndvi=-9999
		cov_ndvi=-9999
		fp_out.attrs['ssd_ndvi']=ssd_ndvi
		fp_out.attrs['cov_ndvi']=cov_ndvi
		continue
	ndvi = fp_ndvi['NDVI'][:]
	lssd_ndvi = []
	lcov_ndvi = []
	for i in range(ndvi.shape[0]):
		ndvi_i = ndvi[i,:,:]
		ndvi_i = ndvi_i[ndvi_i>-9999]
		if len(ndvi_i) ==0:
			continue
		else:
			lcov_ndvi.append(np.std(ndvi_i)/np.mean(ndvi_i))
			lssd_ndvi.append(np.std(ndvi_i))
	ssd_ndvi  =np.mean(lssd_ndvi)
	cov_ndvi  =np.mean(lcov_ndvi)


	# -------------------- #
	# CREATE HDF5 DATASETS #
	# -------------------- #
	fp_out.attrs['ssd_ndvi']=ssd_ndvi
	fp_out.attrs['cov_ndvi']=cov_ndvi


	
	
		
