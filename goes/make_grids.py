import numpy as np
import pickle
import h5py
import rasterio
import os
import sys


grid={'pos_x':[],'pos_y':[],'lon_min':[],'lon_max':[],'lat_max':[],'lat_min':[],'lat_central':[],'lon_central':[]}
gd ='/stor/soteria/hydro/shared/data/GOES/GOES-16-EPSG4326/'
#tif=gd+'2019/'+'OR_ABI-L2-LSTC-M6_G16_s20191831101248_e20191831104021_c20191831104423.tif'
tif=sys.argv[1]
fp=rasterio.open(tif,'r')

in_dir  = '/home/tsw35/soteria/data/eddy_v2/lst/'
for dx in [.25,.5,1]:
    print(dx)
    grid={'site':[],'pos_x':[],'pos_y':[],'lon_min':[],'lon_max':[],'lat_max':[],'lat_min':[],'lat_central':[],'lon_central':[]}
    for file in os.listdir(in_dir):
        print('.',end='',flush=True)
        site=file[0:4]
        fp_in = h5py.File(in_dir+file,'r')
        lat_c=fp_in.attrs['lat']
        lon_c=fp_in.attrs['lon']
        lon_min=lon_c-dx/2
        lon_max=lon_c+dx/2
        lat_min=lat_c-dx/2
        lat_max=lat_c+dx/2
        pos_y,pos_x=fp.index(lon_min,lat_max)
        grid['pos_x'].append(pos_x)
        grid['pos_y'].append(pos_y)
        grid['lon_min'].append(lon_min)
        grid['lon_max'].append(lon_max)
        grid['lat_min'].append(lat_min)
        grid['lat_max'].append(lat_max)
        grid['lat_central'].append(lat_c)
        grid['lon_central'].append(lon_c)
        grid['site'].append(site)
    grid['pos_x'].append(-1)
    grid['pos_y'].append(-1)
    for k in grid.keys():
        if k == 'site':
            continue
        else:
            grid[k]=np.array(grid[k])
            print(grid[k].shape)
    pickle.dump(grid,open("grid"+str(dx)+".p","wb"))



