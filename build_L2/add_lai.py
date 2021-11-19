# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_LAI.py ------------------- #
# ---------------------------------------------------------------- #
# Code Description for add_lai.py

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
lai_dir   = '/home/tsw35/soteria/data/modis/LAI_MCD15A2Hv6/'+\
			'reprojected_MCD15A2Hv6/'
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
	lai = {'tl':[],'lai':[]}
	lai_dirlist = os.listdir(lai_dir)
	lai_dirlist.sort()
	tzutc=datetime.timezone.utc
	for file_lai in lai_dirlist:
		if file_lai[0]=='.':
			continue
		m_date = datetime.datetime(int(file_lai[0:4]),1,1,12,0,\
				 tzinfo=tzutc)+datetime.timedelta(days=int(file_lai[4:7]))
		if(m_date.timestamp()<np.min(time)):
			continue
		fp_lai=rasterio.open(lai_dir+file_lai)
		xy = fp_lai.index(lon,lat)
		try:
			lai_ = fp_lai.read(1)[xy[0],xy[1]]
		except Exception:
			lai_ = -9999
		if lai_<200:
			lai['tl'].append(m_date.timestamp())
			lai['lai'].append(lai_)
		elif lai_>=0:
			lai_=checknearest(fp_lai.read(1),xy[0],xy[1])
			if (lai_>=0)&(lai_<200):
				lai['tl'].append(m_date.timestamp())
				lai['lai'].append(lai_)
	try:
		lai=np.interp(time,lai['tl'],lai['lai'])
	except Exception:
		lai=np.ones((len(time),))*-9999


	# -------------------- #
	# CREATE HDF5 DATASETS #
	# -------------------- #
	try:
		fp_out.create_dataset('LAI',data=np.array(lai))
	except:
		fp_out['LAI'][:]=np.array(lai)
	fp_out['LAI'].attrs['units']='m^2/m^2'
	fp_out['LAI'].attrs['missing_value']=-9999
	fp_out['LAI'].attrs['source']='MODIS: LAI_MCD15A2Hv6'
	print('*',flush=True)
	fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())


	
	
		
