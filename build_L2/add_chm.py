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
import subprocess
import rasterio

from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
neon_dir   = '/home/tsw35/soteria/data/NEON/chm/_tifs/'
neon_dir1  = '/home/tsw35/soteria/data/NEON/chm/'
base_dir   = '../../L2/'
radii=[1000,500,250,100]
canh_avg={}
canh_90={}
canh_75={}
canh_med={}

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir1)
#sites=['ORNL']
for site in sites:
    if site=='ORNL':
        continue
    if len(site)>4:
        continue
    file=site+'_chm.tif'
    print(file,end='...',flush=True)
    canh_avg[file]={}
    canh_90[file]={}
    canh_75[file]={}
    canh_med[file]={}
    fp_L2=h5py.File(base_dir+site+'_L2.h5','r+')
    old_ch=fp_L2.attrs['canopy_height']
    lat=fp_L2.attrs['lat']
    lon=fp_L2.attrs['lon']
    fp_in=rasterio.open(neon_dir+file,'r')
    print('open...',end='',flush=True)
    x,y=fp_in.index(lon,lat)
    data=fp_in.read(1)
    data[data<0]=float('nan')
    print('loaded...',end='',flush=True)
    for r in radii:
        try:
            canh_avg[file][r]=np.nanmean(data[x-r:x+r,y-r:y+r])
            canh_90[file][r]=np.nanpercentile(data[x-r:x+r,y-r:y+r],90)
            canh_75[file][r]=np.nanpercentile(data[x-r:x+r,y-r:y+r],75)
            canh_med[file][r]=np.nanmedian(data[x-r:x+r,y-r:y+r])
        except Exception as e:
            print(e)
            canh_avg[file][r]=old_ch
            canh_90[file][r]=old_ch
            canh_75[file][r]=old_ch
            canh_med[file][r]=old_ch
            
    print(str(x)+','+str(y))
for site in sites:
    if len(site)>4:
        continue
    file=site+'_chm.tif'
    site=file[0:4]
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    old_ch=fp_out.attrs['canopy_height']
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    try:
        fp_out.create_group('canopy_heights')
    except Exception:
        pass
    for r in radii:
        try:
            avg=canh_avg[file][r]
            _90=canh_90[file][r]
            _75=canh_75[file][r]
            med=canh_med[file][r]
        except Exception as e:
            print(e)
            avg=old_ch
            _90=old_ch
            _75=old_ch
            med=old_ch
        if (avg<2) or (np.isnan(avg)):
            avg=old_ch
        if (_90<2)or (np.isnan(_90)):
            _90=old_ch
        if (_75<2)or (np.isnan(_75)):
            _75=old_ch
        if (med<2) or (np.isnan(med)):
            print(site+': '+str(med))
            med=old_ch
        fp_out['canopy_heights'].attrs['avg_radius'+str(r)]=avg
        fp_out['canopy_heights'].attrs['median_radius'+str(r)]=med
        fp_out['canopy_heights'].attrs['90pct_radius'+str(r)]=_90
        fp_out['canopy_heights'].attrs['75pct_radius'+str(r)]=_75
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
print('Exported')


    
    
        
