import os
import subprocess
import datetime
from mpi4py import MPI

# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------------- #
# USER INPUTS and CONSTANTS #
# ------------------------- #
goes_dir = '/stor/soteria/hydro/shared/data/GOES/GOES-16-EPSG4326/'
resos    = ['0.25','0.5','1'] # resolutions to do
out_dir  = '/home/tsw35/tyche/data/GOES/pickles/'

# ------------------- #
# MAIN CODE and SETUP #
# ------------------- #
years=os.listdir(goes_dir)
years.sort()
for year in years:
    if year == '2017':
        year='2017/fixed_resolution'
    f=0
    filelist=os.listdir(goes_dir+year+'/')[rank::size]
    for file in filelist:
        if (f%round(len(filelist)/10)==0):
            print('RANK '+str(rank)+' YEAR '+year+' '+str(f/len(filelist)*100)+'%',flush=True)
        f=f+1
        
        for dx in resos:
            
            cmd= 'python stats.py -d '+dx+' -i '+goes_dir+year+'/'+file+\
                 ' -g '+'grid'+dx+'.p -o '+out_dir+str(dx)+'x'+str(dx)+'_arcdegree/'
            print(cmd)
            subprocess.run(cmd,shell=True)
        


