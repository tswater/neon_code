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
neon_dir   = '/home/tsw35/soteria/data/NEON/soilwater/'
base_dir   = '../../L2/'

out_swc=[]

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
	
	# Load in the base file
	fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
	time=fp_out['TIME'][:]
	out_swc=[]
	old_month=0
	for t in time:
		tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
		if tdt.month!=old_month:
			# Check to see if file exists for this month
			old_month=tdt.month
			filelist=[]
			date_str=str(tdt.year)+'-'+f'{tdt.month:02}'
			for file in os.listdir(neon_dir+site):
				filesplits=file.split('.')
				file_horiz=int(filesplits[6])
				file_vert =int(filesplits[7][1:])
				if (date_str in file)&(file_vert==1):
					filelist.append(file)
			if len(filelist)>0:
				print('.',end='',flush=True)
				_load=True
			else:
				print("'",end='',flush=True)
				_load=False
		else:
			_load=False
		if _load:
			swc_times_all={}
			swc_swc_all={}
			for file in filelist:
				# setup data structure
				if file not in swc_swc_all.keys():
					swc_swc_all[file]=[]
					swc_times_all[file]=[]
				
				# OPEN file and load in info
				with open(neon_dir+site+'/'+file) as read_r:
					read_r = csv.reader(read_r)
					for row in read_r:
						if row[0] == 'startDateTime':
							continue
						ts = row[0]
						dt = datetime.datetime(int(ts[0:4]),int(ts[5:7]),\
							int(ts[8:10]),int(ts[11:13]),int(ts[14:16]))
						tp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
						swc_times_all[file].append(tp)
						try:
							swc_swc_all[file].append(float(row[2]))
						except:
							swc_swc_all[file].append(-9999)
				
		# --------------------- #
		# PERFORM EACH TIMESTEP #
		# --------------------- #
		#get rad indndex
		sums=0
		count=0
		for file in filelist:
			a=np.where(swc_times_all[file][:]==t)[0]
			if len(a)==0:
				pass
			elif swc_swc_all[file][a[0]]==-9999:
				pass
			else:
				sums=sums+(swc_swc_all[file][a[0]])
				count=count+1
		if count==0:
			out_swc.append(-9999)
			continue
			
		# now that we have an index, load the data
		out_swc.append(sums/count)
	# -------------------- #
	# CREATE HDF5 DATASETS #
	# -------------------- #
	try:
		fp_out.create_dataset('SWC',data=np.array(out_swc))
	except:
		fp_out['SWC'][:]=np.array(out_swc)
	fp_out['SWC'].attrs['source']='NEON soil water content'
	fp_out['SWC'].attrs['missing_value']=-9999
	fp_out['SWC'].attrs['description']='Top Layer Soil Water Content'
	fp_out['SWC'].attrs['units']='0-1'
	print('*',flush=True)
	fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())



	
	
		
