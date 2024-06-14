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
deg        = 1
# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
fp_lst=nc.Dataset('/stor/tyche/hydro/shared/GOES_data/compressed_nc/GOES_LST_mean+std+CL+P_1x1degree2_v2.nc','r')
lats=fp_lst['lat'][:]
lons=fp_lst['lon'][:]
times=[]
dt0=datetime.datetime(2017,5,24,0,0)
for t in fp_lst['time'][:]:
    dtl=dt0+datetime.timedelta(hours=t)
    times.append((dtl-datetime.datetime(1970,1,1,0,0)).total_seconds())
print(np.min(times))
print(np.max(times))
for site in sites:
    if not (len(site)==4):
        continue
    if site == 'zips':
        continue
    print(site)
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    timen=fp_out['TIME'][:]
    lat=float(fp_out.attrs['lat'])
    lon=float(fp_out.attrs['lon'])
    lt=np.argmin(np.abs(lats-lat))
    ln=np.argmin(np.abs(lons-lon))
    latdif=np.min(np.abs(lats-lat))
    londif=np.min(np.abs(lons-lon))
    stds=np.interp(timen,times,fp_lst['std'][:,lt,ln],left=-9999,right=-9999)
    stds[stds<0]=-9999
    means=np.interp(timen,times,fp_lst['mean'][:,lt,ln],left=-9999,right=-9999)
    means[means<0]=-9999
    cl5s=np.interp(timen,times,fp_lst['cl_5'][:,lt,ln],left=-9999,right=-9999)
    cl5s[cl5s<0]=-9999
    cl10s=np.interp(timen,times,fp_lst['cl_10'][:,lt,ln],left=-9999,right=-9999)
    cl10s[cl10s<0]=-9999
    cl15s=np.interp(timen,times,fp_lst['cl_15'][:,lt,ln],left=-9999,right=-9999)
    cl15s[cl15s<0]=-9999
    cl20s=np.interp(timen,times,fp_lst['cl_20'][:,lt,ln],left=-9999,right=-9999)
    cl20s[cl20s<0]=-9999
    p25s=np.interp(timen,times,fp_lst['p25'][:,lt,ln],left=-9999,right=-9999)
    p25s[p25s<0]=-9999
    p50s=np.interp(timen,times,fp_lst['p50'][:,lt,ln],left=-9999,right=-9999)
    p50s[p50s<0]=-9999
    p75s=np.interp(timen,times,fp_lst['p75'][:,lt,ln],left=-9999,right=-9999)
    p75s[p75s<0]=-9999
    
    try:
        fp_out.create_dataset('LST_STD',data=np.array(stds))
    except Exception:
        fp_out['LST_STD'][:]=np.array(stds)
    fp_out['LST_STD'].attrs['units']='K'
    fp_out['LST_STD'].attrs['missing_value']=-9999
    fp_out['LST_STD'].attrs['source']='GOES-Laura'
    fp_out['LST_STD'].attrs['latdif']=latdif
    fp_out['LST_STD'].attrs['londif']=londif
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LST_MEAN',data=np.array(means))
    except Exception:
        fp_out['LST_MEAN'][:]=np.array(means)
    fp_out['LST_MEAN'].attrs['units']='K'
    fp_out['LST_MEAN'].attrs['missing_value']=-9999
    fp_out['LST_MEAN'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LST_CL5',data=np.array(cl5s))
    except Exception:
        fp_out['LST_CL5'][:]=np.array(cl5s)
    fp_out['LST_CL5'].attrs['units']='m'
    fp_out['LST_CL5'].attrs['missing_value']=-9999
    fp_out['LST_CL5'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow()) 
     
    try:
        fp_out.create_dataset('LST_CL15',data=np.array(cl15s))
    except Exception:
        fp_out['LST_CL15'][:]=np.array(cl15s)
    fp_out['LST_CL15'].attrs['units']='m'
    fp_out['LST_CL15'].attrs['missing_value']=-9999
    fp_out['LST_CL15'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LST_CL10',data=np.array(cl10s))
    except Exception:
        fp_out['LST_CL10'][:]=np.array(cl10s)
    fp_out['LST_CL10'].attrs['units']='m'
    fp_out['LST_CL10'].attrs['missing_value']=-9999
    fp_out['LST_CL10'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LST_CL20',data=np.array(cl20s))
    except Exception:
        fp_out['LST_CL20'][:]=np.array(cl20s)
    fp_out['LST_CL20'].attrs['units']='m'
    fp_out['LST_CL20'].attrs['missing_value']=-9999
    fp_out['LST_CL20'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LST_P25',data=np.array(p25s))
    except Exception:
        fp_out['LST_P25'][:]=np.array(p25s)
    fp_out['LST_P25'].attrs['units']='K'
    fp_out['LST_P25'].attrs['missing_value']=-9999
    fp_out['LST_P25'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    
    try:
        fp_out.create_dataset('LST_P50',data=np.array(p50s))
    except Exception:
        fp_out['LST_P50'][:]=np.array(p50s)
    fp_out['LST_P50'].attrs['units']='K'
    fp_out['LST_P50'].attrs['missing_value']=-9999
    fp_out['LST_P50'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LST_P75',data=np.array(p75s))
    except Exception:
        fp_out['LST_P75'][:]=np.array(p75s)
    fp_out['LST_P75'].attrs['units']='K'
    fp_out['LST_P75'].attrs['missing_value']=-9999
    fp_out['LST_P75'].attrs['source']='GOES-Laura'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())


