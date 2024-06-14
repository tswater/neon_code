import numpy as np
import os 
import subprocess
import h5py
import datetime

base_dir   = '/home/tsw35/soteria/data/eddy_v2/old/'
sites=  os.listdir(base_dir)

method = 0 # 0 is average 123, 1 is 12, 2 is 23

u1={}
u2={}
u3={}
zd12={}
zd23={}
zd13={}

sites=['ABBY']
for file in sites:
    site=file[0:4]
    print(site)
    fp=h5py.File(base_dir+site+'_L2.h5','r+')
    levels=fp['vertical_wind'].keys()
    top_level=0
    next_level=0
    for k in levels:
        k_lvl=float(k[5:])
        if k_lvl>top_level:
            next_level=top_level
            top_level=k_lvl
    z2=top_level
    z3=next_level
    z1=fp.attrs['tow_height']
    try:
        u2[site]=fp['vertical_wind']['WIND_'+str(top_level)]
    except:
        u2[site]=fp['vertical_wind']['WIND_'+str(top_level)+'0']
    try:
        u3[site]=fp['vertical_wind']['WIND_'+str(next_level)]
    except:
        u3[site]=fp['vertical_wind']['WIND_'+str(next_level)+'0']
    u1[site]=fp['WS'][:]
    k=.41
    ustar=fp['USTAR'][:]
    exp1=np.exp(u1[site][:]*k/ustar)
    exp2=np.exp(u2[site][:]*k/ustar)
    exp3=np.exp(u3[site][:]*k/ustar)
    
    zd12[site]=(z2-z1*exp2/exp1)/(1-exp2/exp1)
    zd13[site]=(z3-z1*exp3/exp1)/(1-exp3/exp1)
    zd23[site]=(z3-z2*exp3/exp2)/(1-exp3/exp2)

    
# ---------------------------- #
# PART 2: SEASONAL CLIMATOLOGY #
# ---------------------------- #
print('Seasonal Climatology...')

# Set up a mask for each season
fp=h5py.File(base_dir+site+'_L2.h5','r')
times=fp['TIME'][:]
masks={'winter':np.zeros((len(times),),dtype=bool),
       'spring':np.zeros((len(times),),dtype=bool),
       'summer':np.zeros((len(times),),dtype=bool),
       'fall':np.zeros((len(times),),dtype=bool)}

for i in range(len(times)):
    t=times[i]
    dt=datetime.datetime.utcfromtimestamp(t)
    if dt.month in [12,1,2]:
        masks['winter'][i]=True
    elif dt.month in [3,4,5]:
        masks['spring'][i]=True
    elif dt.month in [6,7,8]:
        masks['summer'][i]=True
    elif dt.month in [9,10,11]:
        masks['fall'][i]=True
    else:
        print('FAIL')

#### set up the middle of season dates ####
min_date=datetime.datetime.fromtimestamp(times[0],tz=datetime.timezone.utc)
max_date=datetime.datetime.fromtimestamp(times[-1],tz=datetime.timezone.utc)
min_date=min_date-datetime.timedelta(days=70)
max_date=max_date+datetime.timedelta(days=70)
mid_timestamps=[]
cdt=min_date
while cdt<max_date:
    if (cdt.month==1)&(cdt.day==16)&(cdt.hour==12)&(cdt.minute==0):
        mid_timestamps.append(cdt.timestamp())
    elif (cdt.month==4)&(cdt.day==15)&(cdt.hour==23)&(cdt.minute==30):
        mid_timestamps.append(cdt.timestamp())
    elif (cdt.month==7)&(cdt.day==16)&(cdt.hour==12)&(cdt.minute==0):
        mid_timestamps.append(cdt.timestamp())
    elif (cdt.month==11)&(cdt.day==15)&(cdt.hour==23)&(cdt.minute==30):
        mid_timestamps.append(cdt.timestamp())
    cdt=cdt+datetime.timedelta(minutes=30)

from scipy.stats import mode

for file in sites:
    site=file[0:4]
    print(site)
    fp=h5py.File(base_dir+site+'_L2.h5','r+')
    towh=fp.attrs['tow_height']
    s_zd={}
    zd_out=np.ones((len(fp['TIME'][:]),))*-9999

    # determine seasonal averages
    for x in masks.keys():
        zL=fp['ZL'][masks[x]]
        u1_=u1[site][masks[x]]
        u2_=u2[site][masks[x]]
        u3_=u3[site][masks[x]]
        zd12_=zd12[site][masks[x]]
        zd13_=zd13[site][masks[x]]
        zd23_=zd23[site][masks[x]]
        m=(np.abs(zL)<.1)&(u1_>1)
        if method==0:
            # check the first zd12
            ma=m&(zd12_<towh)&(zd12_>-1)&((u1_-u2_)>.05*u1_)
            for i in range(30):
                n=np.sum(ma)
                a_mode=mode(zd12_[ma])
                a_m=a_mode[0][0]
                a_c=a_mode[1][0]
                if a_c>=.01*n:
                    ma=ma&(zd12_!=a_m)
                else:
                    break
            # check zd13
            mb=m&(zd13_<towh)&(zd13_>-1)&((u1_-u3_)>.05*u1_)
            for i in range(30):
                n=np.sum(mb)
                a_mode=mode(zd13_[mb])
                a_m=a_mode[0][0]
                a_c=a_mode[1][0]
                if a_c>=.01*n:
                    mb=mb&(zd13_!=a_m)
                else:
                    break
            # check zd23
            mc=m&(zd23_<towh)&(zd23_>-1)&((u2_-u3_)>.05*u2_)
            for i in range(30):
                n=len(mc)
                a_mode=mode(zd23_[mc])
                a_m=a_mode[0][0]
                a_c=a_mode[1][0]
                if a_c>=.01*n:
                    mc=mc&(zd23_!=a_m)
                else:
                    break
            good=(ma|mb|mc)
            cnt=ma.astype(int)+mb.astype(int)+mc.astype(int)
            s_zd[x]=np.zeros((len(ma),))
            s_zd[x][ma]=s_zd[x][ma]+zd12_[ma]
            s_zd[x][mb]=s_zd[x][mb]+zd13_[mb]
            s_zd[x][mc]=s_zd[x][mc]+zd23_[mc]
            s_zd[x]=s_zd[x]/(cnt+.0001)
            outstuff=s_zd[x][:]
            outstuff[~good]=-9999
            zd_out[masks[x]]=outstuff[:]
            s_zd[x]=np.mean(s_zd[x][good])
        elif method==1:
            ma=m&(zd12_<towh)&(zd12_>-1)&((u1_-u2_)>.05*u1_)
            for i in range(30):
                n=np.sum(ma)
                a_mode=mode(zd12_[ma])
                a_m=a_mode[0][0]
                a_c=a_mode[1][0]
                if a_c>=.01*n:
                    ma=ma&(zd12_!=a_m)
                else:
                    break
            s_zd[x]=np.mean(zd12_[ma])
            outstuff=np.ones((len(zL),))*-9999
            outstuff[ma]=zd12_[ma]
            zd_out[masks[x]]=outstuff
        elif method==2:
            mc=m&(zd23_<towh)&(zd23_>-1)&((u2_-u3_)>.05*u2_)
            for i in range(30):
                n=len(mc)
                a_mode=mode(zd23_[mc])
                a_m=a_mode[0][0]
                a_c=a_mode[1][0]
                if a_c>=.01*n:
                    mc=mc&(zd23_!=a_m)
                else:
                    break
            s_zd[x]=np.mean(zd23_[mc])
            outstuff=np.ones((len(zL),))*-9999
            outstuff[mc]=zd23_[mc]
            zd_out[masks[x]]=outstuff

    # attach them to corresponding mid season times
    mid_values=[]
    for t in mid_timestamps:
        dt=datetime.datetime.fromtimestamp(t,tz=datetime.timezone.utc)
        if dt.month==1:
            mid_values.append(s_zd['winter'])
        elif dt.month==4:
            mid_values.append(s_zd['spring'])
        elif dt.month==7:
            mid_values.append(s_zd['summer'])
        elif dt.month==11:
            mid_values.append(s_zd['fall'])

    # Finally, do the interpolation
    s_zd=np.interp(times,mid_timestamps,mid_values)
    print(str(len(s_zd))+':'+str(len(times)))
    try:
        fp.create_dataset('ZD_SEASONAL',data=np.array(s_zd))
    except:
        fp['ZD_SEASONAL'][:]=np.array(s_zd)
    fp['ZD_SEASONAL'].attrs['missing_value']=-9999
    fp['ZD_SEASONAL'].attrs['source']='derived from ZD2'
    fp['ZD_SEASONAL'].attrs['description']='zero plane displacement height'+\
                                           ' interpolated seasonal averages'
    n=len(s_zd)
    m=fp['ZL']==-9999
    zL2=np.zeros((n,))
    vpt_flux=fp['H'][:]/fp['RHO'][:]/1005
    moL = -fp['USTAR'][:]**3*fp['VPT'][:]/(.4*9.81*vpt_flux)
    zL2=(fp.attrs['tow_height']-s_zd)/moL
    zL2[m]=-9999
    try:
        fp.create_dataset('ZL2',data=np.array(zL2))
    except:
        fp['ZL2'][:]=np.array(zL2)

    fp['ZL2'].attrs['missing_value']=-9999
    fp['ZL2'].attrs['source']='various NEON tow data'
    fp['ZL2'].attrs['description']='MO Stability parameter using'+\
                                   ' ZD_SEASONAL instead of zd'

    
    try:
        fp.create_dataset('ZD2',data=np.array(zd_out))
    except:
        fp['ZD2'][:]=np.array(zd_out)
    fp['ZD2'].attrs['missing_value']=-9999
    fp['ZD2'].attrs['source']='derived from NEON vertical wind profiles'
    fp['ZD2'].attrs['description']='zero plane displacement height derived'+\
                                   ' from vertical canopy wind profiles'
                                   




