import numpy as np
import h5py
import datetime
import os
# add filtered temperature and moisture variance to the L2 files

# --------------- #
# CONSTANTS/SETUP #
# --------------- #
L2_dir   ='../../L2/'
temp_dir ='../../../NEON/filtered_temp/'
h2o_dir  ='../../../NEON/filtered_h2o/'

# -------- #
# CODEBASE #
# -------- #
sites=os.listdir(L2_dir)
#sites=['x-KONA.h5']
for file in sites:
	site=file[0:4]
	print(file)
	fp_L2 =h5py.File(L2_dir+file,'r+')
	fp_h2o=h5py.File(h2o_dir+site+'_hvar.h5','r')
	fp_tv =h5py.File(temp_dir+site+'_tvar.h5','r')
	utcoff=fp_L2.attrs['utc_off']
	print(utcoff)
	ltime=fp_L2['TIME'][:]
	htime=fp_h2o['time'][:]
	ttime=fp_tv['time'][:]
	tv=fp_tv['filtered_var'][:]
	tvc=fp_tv['product_var'][:]
	tss=[]
	tssc=[]
	tti=0
	t_old=0
	for t in ltime:
		if t<t_old:
			tti=0
		t_old=t
		if tti==len(ttime):
			tss.append(-1)
			tssc.append(-1)
		elif np.abs(ttime[tti]-t)<60*5:
			tss.append(tv[tti])
			tssc.append(tvc[tti])
			tti=tti+1
		elif ttime[tti]>t:
			tss.append(-1)
			tssc.append(-1)
		else:
			while ttime[tti]<t:
				tti=tti+1
				if tti==len(ttime):
					tti=tti-1
					break
			if ttime[tti]==t:
				tss.append(tv[tti])
				tssc.append(tvc[tti])
			else:
				tss.append(-1)
				tssc.append(-1)
	hss=[0]
	tss=np.sqrt(np.array(tss))
	tssc=np.sqrt(np.array(tssc))
	print(len(ltime))
	print(len(tss))
	if 'TSS_CHECK' in fp_L2.keys():
		fp_L2['TSS_CHECK'][:]=tssc[:]
	else:
		fp_L2.create_dataset('TSS_CHECK',data=np.array(tssc),fillvalue=-1)
	if 'TSS_FILTER' in fp_L2.keys():
		fp_L2['TSS_FILTER'][:]=tss[:]
	else:
		fp_L2.create_dataset('TSS_FILTER',data=np.array(tss),fillvalue=-1)
	if 'HSS_FILTER' in fp_L2.keys():
		fp_L2['HSS_FILTER'][:]=hss[:]
	else:
		fp_L2.create_dataset('HSS_FILTER',data=np.array(hss),fillvalue=-1)
