# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_VCF.py ------------------- #
# ---------------------------------------------------------------- #
# Code Description for ADD_VCF.py

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
vcf_dir   = '/home/tsw35/soteria/data/modis/VCF_MOD44Bv6/'+\
			'reprojected_MOD44Bv6/'
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
	lat=float(fp_out.attrs['lat'])
	lon=float(fp_out.attrs['lon'])
	
	# Load in the VCF
	vcf = {'tv':[],'tn':[],'tt':[],'noveg':[],'veg':[],'tree':[]}
	vcf_dirlist = os.listdir(vcf_dir)
	vcf_dirlist.sort()
	tzutc=datetime.timezone.utc
	for file_vcf in vcf_dirlist:
		if file_vcf[0]=='.':
			continue
		fp_vcf=rasterio.open(vcf_dir+file_vcf)
		xy = fp_vcf.index(lon,lat)
		m_date = datetime.datetime(int(file_vcf[0:4]),1,1,12,0,\
				 tzinfo=tzutc)+datetime.timedelta(days=64)
		try:
			noveg_ = fp_vcf.read(1)[xy[0],xy[1]]
			veg_   = fp_vcf.read(2)[xy[0],xy[1]]
			tree_  = fp_vcf.read(3)[xy[0],xy[1]]
		except Exception:
			noveg_ = -9999
			veg_   = -9999
			tree_  = -9999
		if noveg_<100:
			vcf['tn'].append(m_date.timestamp())
			vcf['noveg'].append(noveg_)
		elif noveg_>=0:
			noveg_=checknearest(fp_vcf.read(1),xy[0],xy[1])
			if (noveg_>=0)&(noveg_<100):
				vcf['tn'].append(m_date.timestamp())
				vcf['noveg'].append(noveg_)
		if veg_<100:
			vcf['tv'].append(m_date.timestamp())
			vcf['veg'].append(veg_)
		elif veg_>=0:
			veg_=checknearest(fp_vcf.read(2),xy[0],xy[1])
			if (veg_>=0)&(veg_<100):
				vcf['tv'].append(m_date.timestamp())
				vcf['veg'].append(veg_)
		if tree_<100:
			vcf['tt'].append(m_date.timestamp())
			vcf['tree'].append(tree_)
		elif tree_>=0:
			tree_=checknearest(fp_vcf.read(3),xy[0],xy[1])
			if (tree_>=0)&(tree_<100):
				vcf['tt'].append(m_date.timestamp())
				vcf['tree'].append(tree_)
	try:
		bare=np.interp(time,vcf['tn'],vcf['noveg'])
	except Exception:
		bare=np.ones((len(time),))*-9999
	try:
		veg=np.interp(time,vcf['tv'],vcf['veg'])
	except Exception:
		veg=np.ones((len(time),))*-9999
	try:
		tree=np.interp(time,vcf['tn'],vcf['tree'])
	except Exception:
		tree=np.ones((len(time),))*-9999


	# -------------------- #
	# CREATE HDF5 DATASETS #
	# -------------------- #
	try:
		fp_out.create_dataset('BARE_SOIL',data=np.array(bare))
	except:
		fp_out['BARE_SOIL'][:]=np.array(bare)
	fp_out['BARE_SOIL'].attrs['units']='%'
	fp_out['BARE_SOIL'].attrs['missing_value']=-9999
	fp_out['BARE_SOIL'].attrs['source']='MODIS: VCF_MOD44Bv6'
	try:
		fp_out.create_dataset('VEG',data=np.array(veg))
	except Exception:
		fp_out['VEG'][:]=np.array(veg)
	fp_out['VEG'].attrs['units']='%'
	fp_out['VEG'].attrs['missing_value']=-9999
	fp_out['VEG'].attrs['source']='MODIS: VCF_MOD44Bv6'
	try:
		fp_out.create_dataset('TREE',data=np.array(tree))
	except Exception:
		fp_out['TREE'][:]=np.array(tree)
	fp_out['TREE'].attrs['units']='%'
	fp_out['TREE'].attrs['missing_value']=-9999
	fp_out['TREE'].attrs['source']='MODIS: VCF_MOD44Bv6'
	print('*',flush=True)




	
	
		
