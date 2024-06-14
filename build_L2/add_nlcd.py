# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_LAI.py ------------------- #
# ---------------------------------------------------------------- #
# Code Description for add_ndvi.py

import numpy as np
import h5py
import datetime
import os
import csv
import rasterio
import subprocess

from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
base_dir  = '/home/tsw35/soteria/data/eddy_v2/lst/'
neon_dir  = '/home/tsw35/soteria/data/NEON/raw_data/'
nlcd_loc = '/stor/soteria/hydro/private/nc153/data/NLCD/NLCD_2016_Land_Cover_Science_product_L48_20190424.img'

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
#sites=['ABBY']
for site in sites[rank::size]:
     if not(len(site)==4):
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
     nlcd_pct  = {11:0,12:0,21:0,22:0,23:0,24:0,31:0,41:0,42:0,43:0,51:0,52:0,\
                    71:0,72:0,73:0,74:0,81:0,82:0,90:0,95:0}
     nlcd_dom = -9999
     # ------------- #
     # NLCD LAND USE #
     # ------------- #
     gdal_trans = "gdaltransform -s_srs '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' -t_srs '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,-0,-0,0 +units=m +no_defs=True' -output_xy < in_.txt > out_.txt"

     # create in_.txt
     fpin_ = open('in_.txt','w')
     fpin_.write(str(lon)+' '+str(lat))
     fpin_.close()

     # create new output
     subprocess.run(gdal_trans,shell=True)
     str_split=open('out_.txt','r').read().split(' ')

     # figure out center point
     xx = np.round(float(str_split[0]))
     yy = np.round(float(str_split[1]))
     ulx = xx-7*30
     uly = yy+7*30
     llx = xx+7*30
     lly = yy-7*30
     gdalcmd = 'gdal_translate -projwin '+str(ulx)+' '+str(uly)+' '+str(llx)+' '+str(lly)+' '+nlcd_loc+' '+'out2_.tif'
     print(gdalcmd)
     subprocess.run(gdalcmd,shell=True)
     loc_data = rasterio.open('out2_.tif','r').read(1)

     # compute statistics
     count = np.sum(loc_data==loc_data)
     max_class=0
     max_val =0.0
     valsum = 0
     for clas in nlcd_pct.keys():
          val = np.sum(loc_data==clas)/count
          nlcd_pct[clas]=val
          valsum=valsum+val
          if val>max_val:
               max_val = val
               max_class = clas
     nlcd_dom=max_class
     if valsum ==0:
          pass
     elif valsum <1:
          for clas in nlcd_pct.keys():
               nlcd_pct[clas]=nlcd_pct[clas]/valsum

     # clear out files
     subprocess.run('rm out_.txt',shell=True)
     subprocess.run('rm in_.txt',shell=True)
     subprocess.run('rm out2_.tif',shell=True)
     
     # -------------------- #
     # CREATE HDF5 DATASETS #
     # -------------------- #
     fp_out.attrs['nlcd_dom']=nlcd_dom
     for key in nlcd_pct:
          fp_out.attrs['nlcd'+str(key)]=nlcd_pct[key]
     fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
     
     

