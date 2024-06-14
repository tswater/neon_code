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
base_dir   = '../../data/eddy_v2/lst/'

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
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
    in_file_path='/'+site+'/dp0p/data/soni/000_0'+str(d)+'0/'
            
    # Load in the base file
    fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
    time=fp_out['TIME'][:]
    out_uu=[]
    out_uv=[]
    out_uw=[]
    out_vw=[]
    out_vv=[]
    out_ww=[]
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
            raw_u=[]
            raw_v=[]
            raw_w=[]
            
            try:
                ur=fp_in[in_file_path]['veloXaxs'][:]
                vr=fp_in[in_file_path]['veloYaxs'][:]
                wr=fp_in[in_file_path]['veloZaxs'][:]
            except Exception:
                print('Error on '+str(tdt),end='',flush=True)
                out_uu.append(-9999)
                out_uv.append(-9999)
                out_uw.append(-9999)
                out_vw.append(-9999)
                out_vv.append(-9999)
                out_ww.append(-9999)
                
                out_i=out_i+1
                day_index=day_index+1
                continue
        
        # --------------------- #
        # PERFORM EACH TIMESTEP #
        # --------------------- #
        if bad_day:
            count1[0]=count1[0]+1
            out_uu.append(-9999)
            out_uv.append(-9999)
            out_uw.append(-9999)
            out_vw.append(-9999)
            out_vv.append(-9999)
            out_ww.append(-9999)
            
            
            out_i=out_i+1
            day_index=day_index+1
            continue
        # extract high frequency temperature durin 30 min period
        usplit=ur[day_index*36000:(day_index+1)*36000]
        vsplit=vr[day_index*36000:(day_index+1)*36000]
        wsplit=wr[day_index*36000:(day_index+1)*36000]
        
        um=usplit>-50
        vm=vsplit>-50
        wm=wsplit>-50
        uvm=um&vm
        vwm=vm&wm
        uwm=um&wm

        if(np.sum(um)<36000*.1) or (np.sum(vm)<36000*.1) or (np.sum(wm)<36000*.1)\
           or (np.sum(vwm)<36000*.1) or (np.sum(uvm)<36000*.1) or (np.sum(uwm)<36000*.1):
            out_uu.append(-9999)
            out_uv.append(-9999)
            out_uw.append(-9999)
            out_vw.append(-9999)
            out_vv.append(-9999)                
            out_ww.append(-9999)
            
            count1[1]=count1[1]+1

            out_i=out_i+1
            day_index=day_index+1
            continue
        
        out_uu.append(np.var(usplit[um]))
        out_vv.append(np.var(vsplit[vm]))
        out_ww.append(np.var(wsplit[wm]))
        
        out_uv.append(np.mean((usplit[uvm]-np.mean(usplit[uvm]))*(vsplit[uvm]-np.mean(vsplit[uvm]))))
        out_vw.append(np.mean((wsplit[vwm]-np.mean(wsplit[vwm]))*(vsplit[vwm]-np.mean(vsplit[vwm]))))
        out_uw.append(np.mean((usplit[uwm]-np.mean(usplit[uwm]))*(wsplit[uwm]-np.mean(wsplit[uwm]))))

        out_i=out_i+1
        day_index=day_index+1
    # -------------------- #
    # CREATE HDF5 DATASETS #
    # -------------------- #
    try:
        fp_out.create_dataset('UU',data=np.array(out_uu))
    except:
        fp_out['UU'][:]=np.array(out_uu)
    fp_out['UU'].attrs['source']='NEON_raw'
    fp_out['UU'].attrs['missing_value']=-9999
    fp_out['UU'].attrs['description']="u'u' averaged"
    fp_out['UU'].attrs['units']='m2s-2'
    print('*',flush=True)
    print(site+str(count1)+' '+str(np.sum(67200-np.array(count1))))
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('VV',data=np.array(out_vv))
    except:
        fp_out['VV'][:]=np.array(out_vv)
    fp_out['VV'].attrs['source']='NEON_raw'
    fp_out['VV'].attrs['missing_value']=-9999
    fp_out['VV'].attrs['description']="v'v' averaged"
    fp_out['VV'].attrs['units']='m2s-2'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('WW',data=np.array(out_ww))
    except:
        fp_out['WW'][:]=np.array(out_ww)
    fp_out['WW'].attrs['source']='NEON_raw'
    fp_out['WW'].attrs['missing_value']=-9999
    fp_out['WW'].attrs['description']="w'w' averaged"
    fp_out['WW'].attrs['units']='m2s-2'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('UV',data=np.array(out_uv))
    except:
        fp_out['UV'][:]=np.array(out_uv)
    fp_out['UV'].attrs['source']='NEON_raw'
    fp_out['UV'].attrs['missing_value']=-9999
    fp_out['UV'].attrs['description']="u'v' averaged"
    fp_out['UV'].attrs['units']='m2s-2'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('UW',data=np.array(out_uw))
    except:
        fp_out['UW'][:]=np.array(out_uw)
    fp_out['UW'].attrs['source']='NEON_raw'
    fp_out['UW'].attrs['missing_value']=-9999
    fp_out['UW'].attrs['description']="u'w' averaged"
    fp_out['UW'].attrs['units']='m2s-2'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

    try:
        fp_out.create_dataset('VW',data=np.array(out_vw))
    except:
        fp_out['VW'][:]=np.array(out_vw)
    fp_out['VW'].attrs['source']='NEON_raw'
    fp_out['VW'].attrs['missing_value']=-9999
    fp_out['VW'].attrs['description']="v'w' averaged"
    fp_out['VW'].attrs['units']='m2s-2'
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    
    
        
