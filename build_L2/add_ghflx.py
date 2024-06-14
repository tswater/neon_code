# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_RAD.py ------------------- #
# ---------------------------------------------------------------- #
# Add radiaton (netrad) and determine night vs day

import numpy as np
import h5py
import datetime
import os
import csv

from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
neon_dir   = '/home/tsw35/soteria/data/NEON/soil/'
base_dir   = '/home/tsw35/soteria/data/eddy_v2/lst/'

out_ghf=[]

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['LAJA']
for site in sites[rank::size]:
     if len(site)>4:
          continue
     if site == 'zips':
          continue
     print(site+': ',end='',flush=True)
     
     # Load in the base file
     fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
     time=fp_out['TIME'][:]
     out_ghf=[]
     old_month=0
     for t in time:
          tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
          if tdt.month!=old_month:
               # Check to see if file exists for this month
               old_month=tdt.month
               filelist=[]
               date_str=str(tdt.year)+'-'+f'{tdt.month:02}'
               for file in os.listdir(neon_dir+site):
                    if (date_str in file):
                         filelist.append(file)
               if len(filelist)>0:
                    print('.',end='',flush=True)
                    _load=True
               else:
                    print("'",end='',flush=True)
                    _load=False
          else:
               _load=False
          if _load:
               ghf_times_all={}
               ghf_ghf_all={}
               for file in filelist:
                    ghf_times_all[file]=[]
                    ghf_ghf_all[file]=[]
                    with open(neon_dir+site+'/'+file,encoding='latin-1') as read_r:
                         read_r = csv.reader(read_r)
                         for row in read_r:
                              if row[0] == 'startDateTime':
                                   continue
                              ts = row[0]
                              dt = datetime.datetime(int(ts[0:4]),int(ts[5:7]),\
                                   int(ts[8:10]),int(ts[11:13]),int(ts[14:16]))
                              tp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                              ghf_times_all[file].append(tp)
                              try:
                                   ghf_ghf_all[file].append(float(row[2]))
                              except:
                                   ghf_ghf_all[file].append(-9999)
                    
          # --------------------- #
          # PERFORM EACH TIMESTEP #
          # --------------------- #
          #get rad indndex
          sums=0
          count=0
          for file in filelist:
               a=np.where(ghf_times_all[file][:]==t)[0]
               if len(a)==0:
                    pass
               elif ghf_ghf_all[file][a[0]]==-9999:
                    pass
               else:
                    sums=sums+(ghf_ghf_all[file][a[0]])
                    count=count+1
          if count==0:
               out_ghf.append(-9999)
               continue
               
          # now that we have an index, load the data
          out_ghf.append(sums/count)
     # -------------------- #
     # CREATE HDF5 DATASETS #
     # -------------------- #
     try:
          fp_out.create_dataset('G',data=np.array(out_ghf))
     except:
          fp_out['G'][:]=np.array(out_ghf)
     fp_out['G'].attrs['source']='NEON_soilheatflux'
     fp_out['G'].attrs['missing_value']=-9999
     fp_out['G'].attrs['description']='Ground Heat Flux'
     fp_out['G'].attrs['units']='W m-2'
     print('*',flush=True)
     fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())



     
     
          
