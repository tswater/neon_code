# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_RAD.py ------------------- #
# ---------------------------------------------------------------- #
# Add radiaton (netrad) and determine night vs day

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
neon_dir = '/home/tsw35/soteria/data/NEON/netrad/'
base_dir = '../../L2/'
era_dir  = '/home/tsw35/soteria/data/ERA5/h5/'
day_thresh = 50

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['LAJA']
for site in sites[rank::size]:
	if len(site)>4:
		continue
	if site == 'zips':
		continue
	print(site+': ',end='',flush=True)
	
	# Identify the base filename
	site_files=os.listdir(neon_dir+site)
	rad_base=site_files[0][0:52]
	
	# Load in the base file
	fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
	time=fp_out['TIME'][:]
	fp_era5 = h5py.File(era_dir+site+'_era5.h5','r')
	era_cape = fp_era5['CAPE'][:]
	era_tcc  = fp_era5['TCC'][:]
	era_blh  = fp_era5['BLH'][:]
	era_time = fp_era5['TIME'][:]
	out_blh  = []
	out_cape = []
	out_tcc  = []
	for t in time:
		tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
		# --------------------- #
		# PERFORM EACH TIMESTEP #
		# --------------------- #
		
		#get index
		a=np.where(era_time[:]==t)[0]
		if len(a)==0:
			# if this time is not present in rad or file does not
			# exist, we fill with empty
			out_tcc.append(-9999)
			out_cape.append(-9999)
			out_blh.append(-9999)
			continue
		else:
			index=a[0]
		
		# now that we have an index, load the data
		out_tcc.append(era_tcc[index])
		out_blh.append(era_blh[index])
		out_cape.append(era_cape[index])
		
	# -------------------- #
	# CREATE HDF5 DATASETS #
	# -------------------- #
	try:
		fp_out.create_dataset('TCC',data=np.array(out_tcc))
	except:
		fp_out['TCC'][:]=np.array(out_tcc)
	fp_out['TCC'].attrs['missing_value']=-9999
	fp_out['TCC'].attrs['source']='ERA5'
	fp_out['TCC'].attrs['units']='Total Cloud Cover 0-1'
	
	try:
		fp_out.create_dataset('CAPE',data=np.array(out_cape))
	except Exception:
		fp_out['CAPE'][:]=np.array(out_cape)
	fp_out['CAPE'].attrs['units']='J kg-1'
	fp_out['CAPE'].attrs['missing_value']=-9999
	fp_out['CAPE'].attrs['source']='ERA5'
	
	try:
		fp_out.create_dataset('BLH',data=np.array(out_blh))
	except Exception:
		fp_out['BLH'][:]=np.array(out_blh)
	fp_out['BLH'].attrs['units']='m'
	fp_out['BLH'].attrs['missing_value']=-9999
	fp_out['BLH'].attrs['source']='ERA5'
	print('*',flush=True)
	fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())



	
	
		
