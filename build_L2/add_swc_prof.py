# ---------------------------------------------------------------- #
# -------------CODE DESCRIPTION FOR ADD_SWC_PROFpy --------------- #
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
neon_dir   = '/home/tsw35/soteria/data/NEON/soilwater/'
base_dir   = '../../L2/'

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
sites=['CPER','KONZ','TEAK','OAES']
for site in sites[rank::size]:
    if len(site)>4:
        continue
    if site == 'zips':
        continue
    print(site+': ',end='',flush=True)

    # Identify the base filename
    site_files=os.listdir(neon_dir+site)
    base=site_files[0][0:28]

    # Load in the base file
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    time=fp_out['TIME'][:]

    out_swc={}
    out_qf={}
        
    # get a list of horiz and vertical levels
    horiz_levels=[] #FIXME
    horiz_lats=[]
    horiz_lons=[]
    horiz_elev=[]
    vert_levels={}
    vert_locs={}
    
    for file in site_files:
        if 'sensor_positions' in file:
            break
    with open(neon_dir+site+'/'+file) as rr:
        reader=csv.reader(rr)
        for row in reader:
            if row[0] == 'HOR.VER':
                continue
            hrvr=row[0].split('.')
            if hrvr[0] not in horiz_levels:
                try:
                    horiz_lats.append(float(row[15]))
                    horiz_lons.append(float(row[16]))
                    horiz_elev.append(float(row[17]))
                except Exception:
                    continue
                horiz_levels.append(hrvr[0])
                vert_levels[hrvr[0]]=[]
                vert_locs[hrvr[0]]=[]
            if hrvr[1] not in vert_levels[hrvr[0]]:
                vert_levels[hrvr[0]].append(hrvr[1])
                vert_locs[hrvr[0]].append(float(row[11]))
        
    for hlvl in horiz_levels:
        out_qf[hlvl]=np.ones((len(time),len(vert_levels[hlvl])))
        out_swc[hlvl]=np.ones((len(time),len(vert_levels[hlvl])))*-9999
        j=0
        for vlvl in vert_levels[hlvl]:
            if (hlvl==horiz_levels[0])&(vlvl==vert_levels[hlvl][0]):
                pass
            else:
                print('   ',end='',flush=True)
            print(hlvl+':'+vlvl,end='',flush=True)
            old_month=0
            i=0
            for t in time:
                tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
                if tdt.month!=old_month:
                    # check to see if a file exists for this month
                    old_month=tdt.month
                    try:
                        filename=neon_dir+site+'/'+base+hlvl+'.'+vlvl+\
                              '.030.SWS_30_minute.'+str(tdt.year)+'-'+\
                              f'{tdt.month:02}'+'.basic.csv'
                        with open(filename) as read:
                            pass
                        _load=True
                        print('.',end='',flush=True)
                    except Exception as e:
                        _load=False
                        swc_times=[]
                        print("'",end='',flush=True)
                else:
                    _load=False
                if _load:
                    swc_qf=[]
                    swc_swc=[]
                    swc_times=[]
                    with open(filename) as read:
                        read=csv.reader(read)
                        for row in read:
                            if row[0]=='startDateTime':
                                continue
                            ts=row[0]
                            dt=datetime.datetime(int(ts[0:4]),\
                                int(ts[5:7]),int(ts[8:10]),int(ts[11:13])\
                                ,int(ts[14:16]))
                            tp = dt.replace(tzinfo=datetime.timezone.utc)\
                                .timestamp()
                            swc_times.append(tp)
                            swc_qf.append(int(row[9]))
                            try:
                                swc_swc.append(float(row[2]))
                            except:
                                swc_swc.append(-9999)
                # --------------------- #
                # PERFORM EACH TIMESTEP #
                # --------------------- #
                # get index
                a=np.where(swc_times[:]==t)[0]
                if len(a)==0:
                    i=i+1
                    continue
                else:
                    index=a[0]
                out_swc[hlvl][i,j]=swc_swc[index]
                out_qf[hlvl][i,j]=swc_qf[index]
                i=i+1
            print()
            j=j+1
    # -------------------- #
    # CREATE HDF5 DATASETS #
    # -------------------- #
    try:
        fp_out.create_group('SWCs')
    except Exception:
        pass
    i=0
    for hlvl in horiz_levels:
        nm='SWCs/SWC_'+hlvl
        nm2='SWCs/qSWC_'+hlvl
        try:
            fp_out.create_dataset(nm2,data=np.array(out_qf[hlvl][:]))
        except:
            fp_out[nm2][:]=np.array(out_qf[hlvl][:])
        fp_out[nm2].attrs['description']='Quality Flag Values, '+\
                          '1 is failure, 0 is passing'
        try:
            fp_out.create_dataset(nm,data=np.array(out_swc[hlvl][:]))
        except:
            fp_out[nm][:]=np.array(out_swc[hlvl][:])
        fp_out[nm].attrs['source']='NEON soil water content'
        fp_out[nm].attrs['missing_value']=-9999
        fp_out[nm].attrs['description']='Vertical Profile VSoil Water Content'
        fp_out[nm].attrs['lat']=horiz_lats[i]
        fp_out[nm].attrs['lon']=horiz_lons[i]
        fp_out[nm].attrs['elev']=horiz_elev[i]
        fp_out[nm].attrs['depths']=vert_locs[hlvl]
        i=i+1
    print('*',flush=True)
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

