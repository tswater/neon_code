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
    x_len=a3d.shape[1]
    y_len=a3d.shape[2]
    a_out=np.zeros((t_eval,x_len,y_len))
    for x in range(x_len):
        for y in range(y_len):
            col=a3d[:,x,y]
            a_out[:,x,y]=np.interp(t_eval,t,col)
    

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
goes_dir   = '/stor/soteria/hydro/shared/data/GOES/GOES-16-EPSG4326/'
base_dir   = '../L2/'
out_dir    = '../GOES/'

#### LOAD IN LAT/LON ####
lats={}
lons={}
sites=[]
times=[]
for file in os.listdir(base_dir):
    fp_L2=h5py.File(base_dir+file,'r')
    site=file[0:4]
    sites.append(site)
    lats[site]=fp_L2.attrs['lat']
    lons[site]=fp_L2.attrs['lon']
    if len(times) == 0:
        times=fp_L2['TIME'][:]

print('latlons loaded')

#### CREATE EMPTY FILES ####
sites=['ABBY']
for site in sites:
    fp_out=nc.Dataset(out_dir+site+'.nc','w')
    fp_out.createDimension('t',size=len(times))
    fp_out.createVariable('t','d',dimensions=('t'))
    fp_out.name='seconds since 1970,UTC'
    lat=lats[site]
    lon=lons[site]
    times_goes=[]
    LST_goes={}
    dl=0
    i=0
    d=0
    for year in [2017,2018,2019,2020]:
        goes_list=os.listdir(goes_dir+str(year)+'/')
        for file in goes_list:
            if i>=23:
                d=d+1
                i=0
                print(d)
            else:
                i=i+1
            # deal with time
            name_split=file.split('_')
            time_s=name_split[3][1:]
            time_e=name_split[4][1:]
            time_g=(str2timestamp(time_s)+str2timestamp(time_e))/2
            times_goes.append(time_g)
            
            # deal with space
            fp_g=rasterio.open(goes_dir+str(year)+'/'+file)
            if dl==0:
                dl=np.abs(fp_g.xy(0,1)[0]-fp_g.xy(0,0)[0])
                dx=int(round(.5/dl))
            x,y=fp_g.index(lon,lat)
            x=int(round(x))
            y=int(round(y))
            data=fp_g.read(1)
            LST_goes[time_g]=np.array(data[x-dx:x+dx+1,y-dx:y+dx+1])
            lonlist,latlist=fp_g.xy(list(range(x-dx,x+dx+1)),\
                                    list(range(x-dx,x+dx+1)))
    print('data loaded from rasters')
    times_goes.sort()
    LST_goesl=[]
    for t in times_goes:
        LST_goesl.append(LST_goes[t])
    LST_garr=np.array(LST_goesl)
    LST_out=interp3d1x(times,times_goes,LST_garr)
    
    #now, adjust the netcdf4
    fp_out['t'][:]=times
    fp_out.createDimension('lon',size=LST_out.shape[1])
    fp_out.createDimension('lat',size=LST_out.shape[2])
    fp_out.createVariable('lon','d',dimensions=('lon'))
    fp_out['lon'][:]=lonlist[:]
    fp_out.createVariable('lat','d',dimensions=('lat'))
    fp_out['lat'][:]=latlist[:]
    fp_out.createVariable('LST','d',dimensions=('t','lon','lat'))
    fp_out['LST'][:]=LST_out[:]

