# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR GOES.py ----------------------- #
# ---------------------------------------------------------------- #
# Create collocated goes data files for Nate

import numpy as np
import h5py
import datetime
import os
import netCDF4 as nc
import rasterio
from scipy import interpolate

from mpi4py import MPI
# MPI4PY STUFF
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# -------------------- #
# HELPER FUNCTION ---- #
# -------------------- #
def str2timestamp(text):
    year=int(text[0:4])
    days=int(text[4:7])
    hour=int(text[7:9])
    mins=int(text[9:11])
    dt=datetime.datetime(year,1,1,hour,mins,tzinfo=datetime.timezone.utc)\
       +datetime.timedelta(days=days-1)
    return dt.timestamp()

def interp3d1x(t_eval,t_dat,a3d):
    t_dat=np.array(t_dat)
    x_len=a3d.shape[1]
    y_len=a3d.shape[2]
    a_out=np.zeros((len(t_eval),x_len,y_len))
    for x in range(x_len):
        for y in range(y_len):
            col=a3d[:,x,y]
            a_out[:,x,y]=np.interp(t_eval,t_dat,col)
    return a_out

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
goes_dir   = '/stor/soteria/hydro/shared/data/GOES/GOES-16-EPSG4326/'
base_dir   = '../L2/'
out_dir    = '../GOES/'

#### LOAD IN LAT/LON ####
clat={}
clon={}
sites=[]
times=[]
for file in os.listdir(base_dir):
    fp_L2=h5py.File(base_dir+file,'r')
    site=file[0:4]
    sites.append(site)
    clat[site]=fp_L2.attrs['lat']
    clon[site]=fp_L2.attrs['lon']
    if len(times) == 0:
        times=fp_L2['TIME'][:]

print('latlons loaded')

#### CREATE EMPTY FILES ####
goes_datau={}
goes_data={}
lats={}
lons={}
goes_times=[]
dl =0
years=os.listdir(goes_dir)
print('Starting Loading in Raster Data')
for site in sites:
    print(site)
    goes_datau[site]={}
for year in years:
    print(str(year)+':',end='',flush=True)
    i=0
    for file in os.listdir(goes_dir+str(year)+'/'):
        if i>=23:
            print('.',end='',flush=True)
            i=1
        else:
            i=i+1
        # time
        name_split=file.split('_')
        time_s=name_split[3][1:]
        time_e=name_split[4][1:]
        time_g=round((str2timestamp(time_s)+str2timestamp(time_e))/2)
        goes_times.append(int(time_g))
        
        # space
        fp_g=rasterio.open(goes_dir+str(year)+'/'+file)
        if dl==0:
            dl=np.abs(fp_g.xy(0,1)[0]-fp_g.xy(0,0)[0])
            dx=int(round(.5/dl))
        data=fp_g.read(1)
        for site in sites:
            x,y=fp_g.index(clon[site],clat[site])
            x=int(round(x))
            y=int(round(y))
            goes_datau[site][int(time_g)]=np.array(data[x-dx:x+dx+1,y-dx:y+dx+1])
            lons[site],lats[site]=fp_g.xy(list(range(x-dx,x+dx+1)),\
                                          list(range(x-dx,x+dx+1)))
    print()
print('Raster Data Loaded')
goes_times.sort()
for site in sites:
    try:
        print(site)
        goes_data[site]=[]
        for t in goes_times:
            goes_data[site].append(goes_datau[site][int(t)])
        goes_data[site]=np.array(goes_data[site])
        print(goes_data[site].shape)
        site_out=interp3d1x(times,goes_times,goes_data[site])
        print(site_out.shape)
        print()
        fp_out=nc.Dataset(out_dir+site+'.nc','w')
        fp_out.createDimension('t',size=len(times))
        fp_out.createVariable('t','d',dimensions=('t'))
        fp_out.name='seconds since 1970,UTC'
        fp_out['t'][:]=times
        fp_out.createDimension('lon',size=site_out.shape[1])
        fp_out.createDimension('lat',size=site_out.shape[2])
        fp_out.createVariable('lon','d',dimensions=('lon'))
        fp_out['lon'][:]=lons[site][:]
        fp_out.createVariable('lat','d',dimensions=('lat'))
        fp_out['lat'][:]=lats[site][:]
        fp_out.createVariable('LST','d',dimensions=('t','lon','lat'))
        fp_out['LST'][:]=site_out[:]
        fp_out.close()
    except Exception:
        print('SITE '+str(site)+' BAD!!!')





