# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_CLIM.py ------------------- #
# ---------------------------------------------------------------- #
# Add GOES stuff based on Laura's precomputed statistics

import numpy as np
import h5py
import datetime
import os
import csv
import netCDF4 as nc
import pickle

from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
neon_dir   = '/home/tsw35/soteria/data/NEON/raw_data/'
base_dir   = '/home/tsw35/soteria/data/eddy_v2/lst/'
pick_dir   = '/home/tsw35/tyche/data/GOES/pickles/'

dxs = [.25,.5,1]

gvars={'STD':'std','MEAN':'mean','CL5':'cl_5','CL10':'cl_10','CL15':'cl_15',\
       'CL20':'cl_20','P25':'p25','P50':'p50','P75':'p75',\
       'XGRAD':'x_gradient','YGRAD':'y_gradient','XGRAD_ABS':'x_gradient_abs',\
       'YGRAD_ABS':'y_gradient_abs'}

units={'STD':'K','MEAN':'K','CL5':'m','CL10':'m','CL15':'m',\
       'CL20':'m','P25':'K','P50':'K','P75':'K',\
       'XGRAD':'K 2km^-1','YGRAD':'K 2km^-1','XGRAD_ABS':'K 2km^-1',\
       'YGRAD_ABS':'K 2km^-1'}

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
sites.sort()
tfp=h5py.File(base_dir+'ABBY_L2.h5','r')
t_base=tfp['TIME'][:]
tfp.close()
data={}
for site in sites:
    if not (len(site)==4):
        continue
    if site == 'zips':
        continue
    data[site]={}
    for dx in dxs:
        data[site][dx]={}
        for var in gvars.keys():
            data[site][dx][var]=np.ones((len(t_base),))*-9999
print('Loading data...')
y0=2017
for dx in dxs:
    y0=2017
    for t in range(len(t_base)):
        dt=datetime.datetime(1970,1,1,0,0)+datetime.timedelta(seconds=t_base[t])
        pname=pick_dir+str(dx)+'x'+str(dx)+'_arcdegree/cl_'+str(dt)+'p'
        if y0<dt.year:
            print(str(y0)+' is done for '+str(dx),flush=True)
            y0=dt.year
        try:
            maps=pickle.load(open(pname,'rb'))['maps']
            for i in range(len(maps['site'])):
                site=maps['site'][i]
                for var in gvars.keys():
                    data[site][dx][var][t]=maps[gvars[var]][i,i]
        except Exception as e:
            #pname=pick_dir+str(dx)+'x'+str(dx)+'_arcdegree/cl_2019-12-31 10:00:00p'
            #maps=pickle.load(open(pname,'rb'))['maps']
            try:
                pname=pick_dir+str(dx)+'x'+str(dx)+'_arcdegree/cl_'+str(dt-datetime.timedelta(minutes=30))+'p'
                maps=pickle.load(open(pname,'rb'))['maps']
                for i in range(len(maps['site'])):
                    site=maps['site'][i]
                    for var in gvars.keys():
                        data[site][dx][var][t]=maps[gvars[var]][i,i]
            except Exception as e:
                pname=pick_dir+str(dx)+'x'+str(dx)+'_arcdegree/cl_2019-12-31 10:00:00p'
                maps=pickle.load(open(pname,'rb'))['maps']
                for i in range(len(maps['site'])):
                    site=maps['site'][i]
                    for var in gvars.keys():
                        data[site][dx][var][t]=-9999

for site in sites:
    if not (len(site)==4):
        continue
    if site == 'zips':
        continue
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    for var in units.keys():
        for dx in dxs:
            name='LST_'+str(dx)+'_'+var
            try:
                fp_out.create_dataset(name,data=data[site][dx][var][:])
            except Exception:
                fp_out[name][:]=data[site][dx][var][:]
            fp_out[name].attrs['units']=units[var]
            fp_out[name].attrs['missing_value']=-9999
            fp_out[name].attrs['source']='GOES-Laura'
            fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())


