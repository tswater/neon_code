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
neon_dir   = '/home/tsw35/soteria/data/NEON/netrad/'
base_dir   = '../../L2/'
day_thresh = 50

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
    
    # Identify the base filename
    site_files=os.listdir(neon_dir+site)
    rad_base=site_files[0][0:52]
    
    # Load in the base file
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    time=fp_out['TIME'][:]
    out_netrad=[]
    out_swin=[]
    out_swout=[]
    out_lwin=[]
    out_lwout=[]
    out_day=[]
    old_month=0
    for t in time:
        tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
        if tdt.month!=old_month:
            # Check to see if file exists for this month
            old_month=tdt.month
            try:
                filename=neon_dir+site+'/'+rad_base+str(tdt.year)+'-'+\
                          f'{tdt.month:02}'+'.basic.csv'
                with open(filename) as read:
                    pass
                _load=True
                print('.',end='',flush=True)
            except Exception as e:
                _load=False
                rad_times=[]
                mindex=0
                print("'",end='',flush=True)
        else:
            _load=False
        if _load:
            rad_netrad=[]
            rad_times =[]
            rad_swin  =[]
            rad_swout =[]
            rad_lwin = []
            rad_lwout =[]
            with open(filename) as read_r:
                read_r = csv.reader(read_r)
                for row in read_r:
                    if row[0] == 'startDateTime':
                        continue
                    ts = row[0]
                    dt = datetime.datetime(int(ts[0:4]),int(ts[5:7]),\
                        int(ts[8:10]),int(ts[11:13]),int(ts[14:16]))
                    tp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                    rad_times.append(tp)
                    try:
                        netSW = float(row[2]) - float(row[10])
                        netLW = float(row[18]) - float(row[26])
                        rad_netrad.append(netSW+netLW)
                    except:
                        rad_netrad.append(-9999)
                    try:
                        rad_swin.append(float(row[2]))
                    except:
                        rad_swin.append(-9999)
                    try:
                        rad_swout.append(float(row[10]))
                    except:
                        rad_swout.append(-9999)
                    try:
                        rad_lwin.append(float(row[18]))
                    except:
                        rad_lwin.append(-9999)
                    try:
                        rad_lwout.append(float(row[26]))
                    except:
                        rad_lwout.append(-9999)
        # --------------------- #
        # PERFORM EACH TIMESTEP #
        # --------------------- #
        #get rad index
        a=np.where(rad_times[:]==t)[0]
        if len(a)==0:
            # if this time is not present in rad or file does not
            # exist, we fill with empty
            out_swin.append(-9999)
            out_netrad.append(-9999)
            out_day.append(-9999)
            out_swout.append(-9999)
            out_lwout.append(-9999)
            out_lwin.append(-9999)
            continue
        else:
            index=a[0]
        
        # now that we have an index, load the data
        out_swin.append(rad_swin[index])
        out_netrad.append(rad_netrad[index])
        out_swout.append(rad_swout[index])
        out_lwin.append(rad_lwin[index])
        out_lwout.append(rad_lwout[index])
        if np.isnan(rad_swin[index]):
            out_day.append(-9999)
        elif rad_swin[index]<day_thresh:
            out_day.append(0)
        else:
            out_day.append(1)
    # -------------------- #
    # CREATE HDF5 DATASETS #
    # -------------------- #
    try:
        fp_out.create_dataset('DAY',data=np.array(out_day))
    except:
        fp_out['DAY'][:]=np.array(out_day)
    fp_out['DAY'].attrs['missing_value']=-9999
    fp_out['DAY'].attrs['source']='NEON_netrad'
    fp_out['DAY'].attrs['isDaytime']=1
    fp_out['DAY'].attrs['isNighttime']=0
    fp_out['DAY'].attrs['description']='daytime as determined by'+\
                        ' SW_IN with threshold '+str(day_thresh)
    try:
        fp_out.create_dataset('NETRAD',data=np.array(out_netrad))
    except Exception:
        fp_out['NETRAD'][:]=np.array(out_netrad)
    fp_out['NETRAD'].attrs['units']='W m-2'
    fp_out['NETRAD'].attrs['missing_value']=-9999
    fp_out['NETRAD'].attrs['source']='NEON_netrad'
    try:
        fp_out.create_dataset('SW_IN',data=np.array(out_swin))
    except Exception:
        fp_out['SW_IN'][:]=np.array(out_swin)
    fp_out['SW_IN'].attrs['units']='W m-2'
    fp_out['SW_IN'].attrs['missing_value']=-9999
    fp_out['SW_IN'].attrs['source']='NEON_netrad'
    print('*',flush=True)
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('SW_OUT',data=np.array(out_swout))
    except Exception:
        fp_out['SW_OUT'][:]=np.array(out_swout)
    fp_out['SW_OUT'].attrs['units']='W m-2'
    fp_out['SW_OUT'].attrs['missing_value']=-9999
    fp_out['SW_OUT'].attrs['source']='NEON_netrad'
    print('*',flush=True)
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LW_IN',data=np.array(out_lwin))
    except Exception:
        fp_out['LW_IN'][:]=np.array(out_lwin)
    fp_out['LW_IN'].attrs['units']='W m-2'
    fp_out['LW_IN'].attrs['missing_value']=-9999
    fp_out['LW_IN'].attrs['source']='NEON_netrad'
    print('*',flush=True)
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('LW_OUT',data=np.array(out_lwout))
    except Exception:
        fp_out['LW_OUT'][:]=np.array(out_lwout)
    fp_out['LW_OUT'].attrs['units']='W m-2'
    fp_out['LW_OUT'].attrs['missing_value']=-9999
    fp_out['LW_OUT'].attrs['source']='NEON_netrad'
    print('*',flush=True)
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())


    
    
        
