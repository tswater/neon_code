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
	fp_in=h5py.File(neon_dir+site+'/'+os.listdir(neon_dir+site)[0],'r')
	for i in range(10):
		try:
			th=str(i)
			fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
				  '0_30m/tempSoni/']['timeBgn'][:]
			break;
		except:
			pass
	failures=0
	total=0
	for file in os.listdir(neon_dir+site+'/'):
		fp_in=h5py.File(neon_dir+site+'/'+file,'r')
		try:
			pathh='/'+site+'/dp01/data/radiNet/000_0'+th+'0_30m/radiLwOut/'
			radi=fp_in[pathh]['mean'][:]
		except Exception as e:
			failures=failures+1
		total=total+1
	print(site+' failures/total: '+str(failures)+'/'+str(total))
