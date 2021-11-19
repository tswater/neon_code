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
neon_dir   = '/home/tsw35/soteria/data/NEON/precip/'
base_dir   = '../../L2/'

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
	pre_base=site_files[0][0:53]
	
	# Load in the base file
	fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
	time=fp_out['TIME'][:]
	out_precip=[]
	old_month=0
	for t in time:
		tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
		if tdt.month!=old_month:
			# Check to see if file exists for this month
			old_month=tdt.month
			try:
				filename=neon_dir+site+'/'+pre_base+str(tdt.year)+'-'+\
				 		 f'{tdt.month:02}'+'.basic.csv'
				with open(filename) as read:
					pass
				_load=True
				print('.',end='',flush=True)
			except Exception as e:
				_load=False
				pre_times=[]
				print("'",end='',flush=True)
		else:
			_load=False
		if _load:
			pre_times =[]
			pre_precip=[]
			with open(filename) as read_r:
				read_r = csv.reader(read_r)
				for row in read_r:
					if row[0] == 'startDateTime':
						continue
					ts = row[0]
					dt = datetime.datetime(int(ts[0:4]),int(ts[5:7]),\
						int(ts[8:10]),int(ts[11:13]),int(ts[14:16]))
					tp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
					pre_times.append(tp)
					try:
						pre_precip.append(float(row[2])/30/60)
					except:
						pre_precip.append(-9999)
		# --------------------- #
		# PERFORM EACH TIMESTEP #
		# --------------------- #
		#get rad indndex
		a=np.where(pre_times[:]==t)[0]
		if len(a)==0:
			# if this time is not present in rad or file does not
			# exist, we fill with empty
			out_precip.append(-9999)
			continue
		else:
			index=a[0]
			
		# now that we have an index, load the data
		out_precip.append(pre_precip[index])
	# -------------------- #
	# CREATE HDF5 DATASETS #
	# -------------------- #
	try:
		fp_out.create_dataset('P',data=np.array(out_precip))
	except:
		fp_out['P'][:]=np.array(out_precip)
	fp_out['P'].attrs['source']='NEON_precip'
	fp_out['P'].attrs['missing_value']=-9999
	fp_out['P'].attrs['description']='Precipitation'
	fp_out['P'].attrs['units']='mm s-1'
	print('*',flush=True)
	fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())



	
	
		
