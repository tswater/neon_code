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
neon_dir   = '/home/tsw35/soteria/data/NEON/raw_data/'
base_dir   = '../../L2/'

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['ABBY']
count1=[0,0,0,0]
for site in sites[rank::size]:
    if len(site)>4:
        continue
    if site == 'zips':
        continue
    print(site+': ',end='',flush=True)
    
    # Identify the base filename
    site_files=os.listdir(neon_dir+site)
    raw_base=site_files[0][0:33]
    for i in range(10):
        fp=h5py.File(neon_dir+site+'/'+site_files[0])
        try:
            in_file_path='/'+site+'/dp0p/data/soni/000_0'+str(i)+'0/'
            fp[in_file_path]['tempSoni']
            d=i
        except Exception:
            pass
    in_file_path_T='/'+site+'/dp0p/data/soni/000_0'+str(d)+'0/'
    in_file_path_q='/'+site+'/dp0p/data/irgaTurb/000_0'+str(d)+'0/'

    # Load in the base file
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    time=fp_out['TIME'][:]
    out_tq=[]
    out_tqf=[]
    old_day=0
    out_i=0
    line_length=0
    for t in time:
        tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
        if tdt.day!=old_day:
            # Check to see if file exists for this month
            old_day=tdt.day
            day_index=0
            try:
                filename=neon_dir+site+'/'+raw_base+str(tdt.year)+'-'+\
                          f'{tdt.month:02}'+'-'+f'{tdt.day:02}'+\
                         '.l0p.h5'
                fp_in=h5py.File(filename,'r')
                _load=True
                if line_length>60:
                    print('\n   '+'   ',end='',flush=True)
                    line_length=0
                print('.',end='',flush=True)
                bad_day=False
                line_length=line_length+1
            except Exception as e:
                _load=False
                bad_day=True
                if line_length>60:
                    print('\n   '+'   ',end='',flush=True)
                    line_length=0
                print("'",end='',flush=True)
                line_length=line_length+1
        else:
            _load=False
        if _load:
            raw_tssf=[]
            try:
                temprt=fp_in[in_file_path_T]['tempSoni'][:]
                temprq=fp_in[in_file_path_q]['rtioMoleDryH2o'][:]*1000
            except Exception:
                print('Error on '+str(tdt),end='',flush=True)
                out_tqf.append(-9999)
                out_tq.append(-9999)
                out_i=out_i+1
                day_index=day_index+1
                continue
        
        # --------------------- #
        # PERFORM EACH TIMESTEP #
        # --------------------- #
        if bad_day:
            count1[0]=count1[0]+1
            out_tqf.append(-9999)
            out_tq.append(-9999)
            out_i=out_i+1
            day_index=day_index+1
            continue
        # extract 30min high frequency temperature and moisture
        temptsplit=temprt[day_index*36000:(day_index+1)*36000]
        tempqsplit=temprq[day_index*36000:(day_index+1)*36000]

        # unfiltered unadjusted variance
        # varr=np.nanvar(tempsplit)
        # varp=tss[out_i]**2
        # if(varp<0):
        #     count1[1]=count1[1]+1
        #    out_tssf.append(-9999)
        #    out_i=out_i+1
        #    day_index=day_index+1
        #    continue
            
        # Basic Stuff First; remove extremes
        olen=len(temptsplit)
        
        #tmask=(temptsplit>250)&(temptsplit<335)&\
        #      (tempqsplit>=0.00)&(tempqsplit<.15)
        #tmask=tmask&(~np.isnan(temptsplit))&(~np.isnan(tempqsplit))
        tmask=(~np.isnan(temptsplit))&(~np.isnan(tempqsplit))
        temptsplit=temptsplit[tmask]
        tempqsplit=tempqsplit[tmask]
        

        # if there are sufficiently few points left, remove
        if len(temptsplit)<=.01*olen:
            out_tq.append(-9999)
            out_tqf.append(-9999)
            out_i=out_i+1
            count1[2]=count1[2]+1
            day_index=day_index+1
            continue
            
        #### Add unfiltered ####
        out_tq.append(np.cov(temptsplit-np.mean(temptsplit),tempqsplit-np.mean(tempqsplit))[0][1])

        #### Apply Filter ####
        try:
            fftrt=np.fft.rfft(temptsplit-np.nanmean(temptsplit))
            fftrq=np.fft.rfft(tempqsplit-np.nanmean(tempqsplit))
        except Exception:
            count1[3]=count1[3]+1
            out_tqf.append(-9999)
            out_i=out_i+1
            day_index=day_index+1
            continue

        cut = 6
        fftrt[0:cut+1]=0
        fftrq[0:cut+1]=0
        f_t=np.fft.irfft(fftrt)
        f_q=np.fft.irfft(fftrq)
        qtf=np.cov(f_t-np.mean(f_t),f_q-np.mean(f_q))[0][1]
        out_tqf.append(qtf)
        out_i=out_i+1
        day_index=day_index+1
    
    
    # -------------------- #
    # CREATE HDF5 DATASETS #
    # -------------------- #
    
    try:
       fp_out.create_dataset('TQ',data=np.array(out_tq))
    except:
       fp_out['TQ'][:]=np.array(out_tq)
    fp_out['TQ'].attrs['source']='NEON_raw'
    fp_out['TQ'].attrs['missing_value']=-9999
    fp_out['TQ'].attrs['description']='TQ adjusted '+\
                  'over the 30 min period '
    fp_out['TQ'].attrs['units']=''

    try:
        fp_out.create_dataset('TQ_FILTER',data=np.array(out_tqf))
    except:
        fp_out['TQ_FILTER'][:]=np.array(out_tqf)
    fp_out['TQ_FILTER'].attrs['source']='NEON_raw'
    fp_out['TQ_FILTER'].attrs['missing_value']=-9999
    fp_out['TQ_FILTER'].attrs['description']='TQ adjusted for '+\
                      'non-stationarity over the 30 min period '+\
                      'low pass filter removed trends w/ period >5min'
    fp_out['TQ_FILTER'].attrs['units']=''
    print('*',flush=True)
    print(site+str(count1)+' '+str(np.sum(67200-np.array(count1))))
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())


    
    
        
