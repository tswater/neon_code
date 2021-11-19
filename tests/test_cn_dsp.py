import os
import netCDF4 as nc
import numpy as np
import h5py
import datetime
import ephem
import rasterio
import csv
import subprocess

neon_dir = '/home/tsw35/soteria/data/NEON/dp04/'
sites=os.listdir(neon_dir)
for site in sites:
	if len(site)>4:
		continue
	canopy=[]
	disp=[]
	tow=[]
	for file in os.listdir(neon_dir+site+'/'):
		fp_in=h5py.File(neon_dir+site+'/'+file,'r')
		try:
			canopy.append(float(fp_in[site].attrs['DistZaxsCnpy'][0]))
		except:
			canopy.append(-1)
		try:
			disp.append(float(fp_in[site].attrs['DistZaxsDisp'][0]))
		except:
			disp.append(-1)
		try:
			tow.append(float(fp_in[site].attrs['DistZaxsTow'][0]))
		except:
			tow.append(-1)
	good=True
	if (np.min(canopy)!=np.max(canopy)):
		good=False
		print(site+' canopy change: '+str(canopy))
	if (np.min(disp)!=np.max(disp)):
		good=False
		print(site+' disp change: '+str(disp))
	if (np.min(tow)!=np.max(tow)):
		good=False
		print(site+' tow change: '+str(tow))
	if (np.mean(canopy)==-1):
		good=False
		print(site+' no canopy height reported')
	if (np.mean(disp)==-1):
		good=False
		print(site+' no displacement height reported')
	if (np.mean(tow)==-1):
		good=False
		print(site+' no tower height reported')
	if (good):
		print(site+' constant')
