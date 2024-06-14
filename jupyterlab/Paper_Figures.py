# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import cartopy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.ensemble
from sklearn import metrics
import h5py
import datetime
import seaborn as sns
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
sns.set_theme()

# %%
try:
    fp.close()
except:
    pass

# %%
# --------------------------- #
# Setup for Figures and Stuff #
# --------------------------- #
# Load File
fp = h5py.File('data/QAQCv4_neon_pntEB0.2_day_rain_frez_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5','r')
# Sites
focus_sites=[b'WREF',b'NOGP',b'BART',b'SOAP','MAP',b'ORNL',b'SRER',b'KONZ',b'DSNY']
#focus_sites=[b'ABBY',b'SCBI',b'HARV',b'JERC','MAP',b'ORNL',b'TREE',b'OSBS',b'UNDE']
fpsites=fp['site'][:]
lats=fp['lat'][:]
lons=fp['lon'][:]
sites = []
nlat=[]
nlon=[]
fnlat=[]
fnlon=[]
for i in range(len(fpsites)):
    x=fpsites[i]
    if x not in sites:
        nlat.append(lats[i])
        nlon.append(lons[i])
        sites.append(x)
        if x in focus_sites:
            fnlat.append(lats[i])
            fnlon.append(lons[i])

# %%
# ------------------------------------ #
# Figure 1. Covariates vs the Variance #
# ------------------------------------ #
# SETUP
pred = ['H','USTAR','WS','VEG','TREE','H2O','T_SONIC','BARE_SOIL','LAI','ZL','BLH','TCC','CAPE','SWC']
pred = ['H','ZL2','USTAR','WS','T_SONIC','LAI','H2O','BLH','SWC','VEG','TREE','BARE_SOIL','TCC','CAPE']
pred = ['H','ZL2','USTAR','WS','CD','T_SONIC','LAI','BLH','CAPE','TCC','TREE','BARE_SOIL','VEG']
pred = ['H','ZL2','USTAR','WS','CD','T_SONIC','LAI','BLH','CAPE','TCC','TREE','BARE_SOIL','VEG']

names = {'H':'H',
         'LE':'LE',
         'VPD':'VPD',
         'USTAR':'$u_*$',
         'T_SONIC':'$T_{air}$',
         'WS':'$u\bar$',
         'BLH':'BLH',
         'VEG':'$f_{veg}$',
         'LE':'LE',
         'TREE':'$f_{tree}$',
         'BARE_SOIL':'$f_{bare}$',
         'ZL2':'$\zeta$',
         'LAI':'LAI',
         'TCC':'$f_{cloud}$',
         'CAPE':'CAPE',
         'tow_height':'$h_{tower}$',
         'canopy_height':'$h_{canopy}$',
         'PA':'P',
         'H2O':'$C_{H_{2}O}$',
         'SWC':'SWC',
         'CD':'$C_d$'}
xlabels= {'H':'($W/{m^2}$)','USTAR':'($m/s$)','WS':'($m/s$)',
          'H2O':'($mmol_{H2O}/{mol}$)','T_SONIC':'($^\circ$C)',
          'LAI':'($m^2/m^2$)','ZL2':'','BLH':'(m)','SWC':'($m^3/m^3)$',
          'CD':'','LE':'($W/{m^2}$)','CAPE':'(J/kg)',
          'TCC':'(%)','TREE':'(%)','BARE_SOIL':'(%)','VEG':'(%)'}
n = len(fp['H'][:])
n2 = len(pred)
result_old=(fp['T_SONIC_SIGMA'][:])
result=(fp['TSS_FILTER'][:])
mins={'WS':0,'ZL2':-3,'USTAR':0,'CAPE':10,'H':-150,'BLH':0,'TCC':.01,'CD':0}
maxs={'WS':15,'ZL2':1,'USTAR':1.75,'CAPE':1000,'H':650,'BLH':5000,'TCC':.99,'CD':.2}
rmin = 0
rmax = 1.275

# %%
i=1
zL=fp['ZL2'][:]
plt.figure(figsize=(7,7))
for f in focus_sites:
    if f == 'MAP':
        i=i+1
        continue
    m=(f==fpsites)#&(fp['H'][:]>250)&(fp['H'][:]<275)
    plt.subplot(3,3,i)
    plt.hist(result[m],bins=np.linspace(0,1,50))
    plt.title(f)
    i = i+1
plt.subplots_adjust(wspace=.2,hspace=.4)

# %%
i=1
zL=fp['ZL2'][:]
plt.figure(figsize=(7,7))
for f in focus_sites:
    if f == 'MAP':
        i=i+1
        continue
    m=(f==fpsites)&(fp['NETRAD'][:]>350)&(fp['NETRAD'][:]<400)
    plt.subplot(3,3,i)
    plt.hist(result[m],bins=np.linspace(0,1,50))
    plt.title(f)
    i = i+1
plt.subplots_adjust(wspace=.2,hspace=.4)

# %%
plt.hexbin(fp['LE'],result,gridsize=150,cmap='terrain',mincnt=1,extent=(0,600,0,1.25))

# %%
# ------------------------------------ #
# Figure 1. Covariates vs the Variance #
# ------------------------------------ #
# FIGURE
data = np.zeros((n2,n))
i = 0
fig = plt.figure(figsize=(12,9))
for x in pred:
    data[i,:]=fp[x][:]
    if x == 'LAI':
        data[i,:]=data[i,:]/10
    if i>=12:
        continue
    plt.subplot(3,4,i+1)
    try:
        plt.hexbin(data[i,:],result,gridsize=150,cmap='terrain',mincnt=1,extent=(mins[x],maxs[x],rmin,rmax))
    except Exception:
        plt.hexbin(data[i,:],result,gridsize=150,cmap='terrain',mincnt=1)
    #plt.title(names[x],fontsize=12)
    #plt.xlabel(xlabels[x],fontsize=10)
    plt.title(names[x]+' '+xlabels[x],fontsize=12)
    i = i+1
#fig.suptitle('Direct Comparison of Environmental Covariates vs $\sigma_T$',fontsize=20)
data = np.transpose(data)
fig.text(0.05, 0.5, "$\sigma_T$",fontsize=12, va='center', rotation='vertical')
plt.subplots_adjust(hspace=.35,wspace=.3)
plt.show()
fig.savefig('fig1.jpg')

# %%
# ------------------------- #
# Figure 2a. Show Filtering #
# ------------------------- #
basename='../data/NEON/raw_data/ABBY/NEON.D16.ABBY.IP0.00200.001.ecte.2018-07-'
in_file_path='/ABBY/dp0p/data/soni/000_050/'
temp_raw=[]
time_raw=[]
for i in range(1,2):
    if i<10:
        name=basename+'0'+str(i)+'.l0p.h5'
    else:
        name=basename+str(i)+'.l0p.h5'
    print(name)
    fp2=h5py.File(name,'r')
    temp_raw.extend(fp2[in_file_path]['tempSoni'][:])
    time_raw.extend(fp2[in_file_path]['time'][:])
rTemp=np.array(temp_raw)-273

st=15000+5*(20*60*30)
extra=20*60*2
tempslice=rTemp[st-extra:st+20*60*30+extra]
times=np.linspace(0,30,20*60*30)
out3=np.fft.rfft(tempslice-np.mean(tempslice))
out4=out3.copy()
cut=6
out4[0:cut+1]=0
old=np.fft.irfft(out3)
new=np.fft.irfft(out4)
meantimes=[]
meanold=[]
meannew=[]
for i in range(60):
    meantimes.append(times[int((i+.5)*10*60)])
    meanold.append(np.mean(old[i*10*60+extra:(i+1)*10*60+extra]))
    meannew.append(np.mean(new[i*10*60+extra:(i+1)*10*60+extra]))
fig = plt.figure(figsize=(9,9))
gs = fig.add_gridspec(3,2)
ax11 = fig.add_subplot(gs[0,0])
ax12 = fig.add_subplot(gs[0,1])
ax21 = fig.add_subplot(gs[1,0])
ax22 = fig.add_subplot(gs[1,1])
ax3  = fig.add_subplot(gs[2,:])

ax21.plot(times,old[extra:-extra],':',alpha=.5,color='red')
ax21.plot(meantimes,meanold,color='red')
ax21.set_ylim(-.35,.45)
ax21.set_title('Original Data 1 Run   $\sigma_T$: '+f'{np.std(old):1.3f}')
ax21.set_xlabel('Time: Minutes')
ax21.set_ylabel("T'")
ax21.legend(["20Hz data","1 min avg"])

ax22.plot(times,new[extra:-extra],':',alpha=.5,color='darkorange')
ax22.plot(meantimes,meannew,color='darkorange')
ax22.set_title('Filtered Data  1 Run   $\sigma_T$: '+f'{np.std(new):1.3f}')
ax22.set_xlabel('Time: Minutes')
ax22.set_ylim(-.35,.45)
ax22.set_xlim(times[extra],times[-extra])
ax22.legend(["20Hz data","1 min avg"])
print(np.var(np.fft.irfft(out3)))
print(np.sqrt(np.var(np.fft.irfft(out4)[extra:-extra])))

# --------------------------------------- #
# Figure 2b. Show Filtering but nonlinear #
# --------------------------------------- #
basename='../data/NEON/raw_data/ABBY/NEON.D16.ABBY.IP0.00200.001.ecte.2018-07-'
in_file_path='/ABBY/dp0p/data/soni/000_050/'
temp_raw=[]
time_raw=[]
for i in range(1,2):
    if i<10:
        name=basename+'0'+str(i)+'.l0p.h5'
    else:
        name=basename+str(i)+'.l0p.h5'
    print(name)
    fp2=h5py.File(name,'r')
    temp_raw.extend(fp2[in_file_path]['tempSoni'][:])
    time_raw.extend(fp2[in_file_path]['time'][:])
rTemp=np.array(temp_raw)-273

#19 maybe
#23 maybe
#24 maybe

st=round(1000+23.6*(20*60*30))
extra=5 #20*30*2
tempslice=rTemp[st-extra:st+20*60*30+extra]
times=np.linspace(0,30,20*60*30)
out3=np.fft.rfft(tempslice-np.mean(tempslice))
out4=out3.copy()
cut=6
out4[0:cut+1]=0
old=np.fft.irfft(out3)
new=np.fft.irfft(out4)
meantimes=[]
meanold=[]
meannew=[]
for i in range(60):
    meantimes.append(times[int((i+.5)*10*60)])
    meanold.append(np.mean(old[i*10*60+extra:(i+1)*10*60+extra]))
    meannew.append(np.mean(new[i*10*60+extra:(i+1)*10*60+extra]))

ax11.plot(times,old[extra:-extra],':',alpha=.5,color='red')
ax11.plot(meantimes,meanold,color='red')
ax11.set_ylim(-.3,.25)
ax11.set_title('Original Data 1 Run   $\sigma_T$: '+f'{np.std(old):1.3f}')
ax11.set_ylabel("T'")
ax11.legend(["20Hz data","1 min avg"])


ax12.plot(times,new[extra:-extra],':',alpha=.5,color='darkorange')
ax12.plot(meantimes,meannew,color='darkorange')
ax12.set_title('Filtered Data  1 Run   $\sigma_T$: '+f'{np.std(new):1.3f}')
ax12.set_ylim(-.3,.25)
ax12.set_xlim(times[extra],times[-extra])
ax12.legend(["20Hz data","1 min avg"])
print(np.var(np.fft.irfft(out3)))
print(np.var(np.fft.irfft(out4)[extra:-extra]))
print(time_raw[st])


# ------------------------- #
# Figure 2c. Show Filtering #
# ------------------------- #
fpl=h5py.File('../data/eddy_v2/L2/ABBY_L2.h5','r')
st_ind=32128
en_ind=33567
time=fpl['TIME'][st_ind:en_ind]
plottime=[]
for t in time:
    plottime.append(datetime.datetime.utcfromtimestamp(t)+datetime.timedelta(hours=int(fpl.attrs['utc_off'])))
TSS=fpl['T_SONIC_SIGMA'][st_ind:en_ind]
TSSf=fpl['TSS_FILTER'][st_ind:en_ind]
m=(TSS<0)|(TSS>2)|(TSSf<0)|(TSSf>2)
TSS[m]=float('nan')
TSSf[m]=float('nan')
ax3.plot(plottime[0:-1],TSS[0:-1],c='red')
ax3.plot(plottime[0:-1],TSSf[0:-1],c='darkorange')
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b-%d')
#plt.title('Comparing Product and Filtered Temperature Standard Deviation 1 Month ABBY',fontsize=14)
#plt.plot(np.array(out_var)-np.array(out_varp))
plt.legend(['NEON product $\sigma_T$','Filtered $\sigma_T$'])
plt.xlabel('Time')
plt.ylabel('$\sigma_T$')
ax3.set_xticklabels(ax3.get_xticks(), rotation = 15)
ax3.xaxis.set_major_formatter(myFmt)
plt.subplots_adjust(wspace=.2,hspace=.4)

# %%
# --------------------------------------- #
# Figure 2b. Show Filtering but nonlinear #
# --------------------------------------- #
basename='../data/NEON/raw_data/ABBY/NEON.D16.ABBY.IP0.00200.001.ecte.2018-07-'
in_file_path='/ABBY/dp0p/data/soni/000_050/'
temp_raw=[]
time_raw=[]
for i in range(1,2):
    if i<10:
        name=basename+'0'+str(i)+'.l0p.h5'
    else:
        name=basename+str(i)+'.l0p.h5'
    print(name)
    fp2=h5py.File(name,'r')
    temp_raw.extend(fp2[in_file_path]['tempSoni'][:])
    time_raw.extend(fp2[in_file_path]['time'][:])
rTemp=np.array(temp_raw)-273

#19 maybe
#23 maybe
#24 maybe

st=round(1000+23.6*(20*60*30))
extra=5 #20*30*2
tempslice=rTemp[st-extra:st+20*60*30+extra]
times=np.linspace(0,30,20*60*30)
out3=np.fft.rfft(tempslice-np.mean(tempslice))
out4=out3.copy()
cut=6
out4[0:cut+1]=0
old=np.fft.irfft(out3)
new=np.fft.irfft(out4)
meantimes=[]
meanold=[]
meannew=[]
for i in range(60):
    meantimes.append(times[int((i+.5)*10*60)])
    meanold.append(np.mean(old[i*10*60+extra:(i+1)*10*60+extra]))
    meannew.append(np.mean(new[i*10*60+extra:(i+1)*10*60+extra]))
plt.figure(figsize=(9,3))
plt.subplot(1,2,1)
plt.plot(times,old[extra:-extra],':',alpha=.5,color='red')
plt.plot(meantimes,meanold,color='red')
plt.ylim(-.3,.25)
plt.title('Original Data 1 Run   $\sigma_T$: '+f'{np.std(old):1.3f}')
plt.xlabel('Time: Minutes')
plt.ylabel("T'")
plt.legend(["20Hz data","1 min avg"])
plt.subplot(1,2,2)
plt.plot(times,new[extra:-extra],':',alpha=.5,color='darkorange')
plt.plot(meantimes,meannew,color='darkorange')
plt.title('Filtered Data  1 Run   $\sigma_T$: '+f'{np.std(new):1.3f}')
plt.xlabel('Time: Minutes')
plt.ylim(-.3,.25)
plt.xlim(times[extra],times[-extra])
plt.ylabel("T'")
plt.legend(["20Hz data","1 min avg"])
print(np.var(np.fft.irfft(out3)))
print(np.var(np.fft.irfft(out4)[extra:-extra]))
print(time_raw[st])




# %%
len(meantimes)

# %%
# ------------------------- #
# Figure 2c. Show Filtering #
# ------------------------- #
fpl=h5py.File('../data/eddy_v2/L2/ABBY_L2.h5','r')
st_ind=32128
en_ind=33567
time=fpl['TIME'][st_ind:en_ind]
plottime=[]
for t in time:
    plottime.append(datetime.datetime.utcfromtimestamp(t)+datetime.timedelta(hours=int(fpl.attrs['utc_off'])))
TSS=fpl['T_SONIC_SIGMA'][st_ind:en_ind]
TSSf=fpl['TSS_FILTER'][st_ind:en_ind]
m=(TSS<0)|(TSS>2)|(TSSf<0)|(TSSf>2)
TSS[m]=float('nan')
TSSf[m]=float('nan')
fig,ax=plt.subplots(figsize=(9,3))
ax.plot(plottime[0:-1],TSS[0:-1],c='red')
ax.plot(plottime[0:-1],TSSf[0:-1],c='darkorange')
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(myFmt)
#plt.title('Comparing Product and Filtered Temperature Standard Deviation 1 Month ABBY',fontsize=14)
#plt.plot(np.array(out_var)-np.array(out_varp))
plt.legend(['NEON product $\sigma_T$','Filtered $\sigma_T$'])
plt.xlabel('Time')
plt.ylabel('$\sigma_T$')
plt.xticks(rotation=15)


# %%

# %%

# %%
# ------------------------ #
# Figure 3. Diurnal Cycles #
# ------------------------ #
#### FUNCTION #####
def shift(arr,off):
    n=arr.shape[0]
    out=np.zeros((n,))
    k=int(round(off))
    for l in range(n):
        if k>=n:
            k=0
        out[k]=arr[l]
        k=k+1
    return out
        
fig=plt.figure(figsize=(13,11))
i=0
for site in focus_sites:
    if site=='MAP':
        ax= plt.subplot(3,3,i+1,projection=cartopy.crs.PlateCarree())
        ax.background_img('ne_shaded','low')
        ax.set_xlim(np.min(nlon)-3,np.max(nlon)+1)
        ax.set_ylim(np.min(nlat)-1,np.max(nlat)+1)
        ax.add_feature(cartopy.feature.COASTLINE)
        #ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.BORDERS)
        #ax.add_feature(cartopy.feature.LAKES)
        ax.set_aspect('auto')
        plt.scatter(nlon,nlat,c='g',s=50,alpha=.5)
        plt.scatter(fnlon,fnlat,marker='*',c='y',s=200)
        ax.axis('off')
        ax.outline_patch.set_edgecolor('white')
        i=i+1
        continue
    site_name=str(site)[2:-1]
    fpl=h5py.File('../data/eddy_v2/L2/'+site_name+'_L2.h5','r')
    Hl=fpl['H'][:]
    TSSl=fpl['T_SONIC_SIGMA'][:]
    fTSSl=fpl['TSS_FILTER'][:]
    m1=(fpl['qT_SONIC'][:]==1)|(fpl['ZL'][:]>0)
    Hl[m1]=-9999
    TSSl[m1]=-9999
    fTSSl[m1]=-9999
    utcoff=fpl.attrs['utc_off']
    fTSSd=np.zeros((48,))
    c_fTSSd=np.zeros((48,))
    TSSd=np.zeros((48,))
    c_TSSd=np.zeros((48,))
    Hd=np.zeros((48,))
    c_Hd=np.zeros((48,))
    for j in range(48):
        # Sensible Heat Diurnal
        Hmask=(Hl[j::48]>-9999)&(Hl[j::48]<1200)
        c_Hd[j]=np.sum(Hmask)
        Hd[j]=np.sum(Hl[j::48][Hmask])
        
        # Filtered TSS Diurnal
        fTSSmask=(fTSSl[j::48]>0)&(fTSSl[j::48]<1.5)
        c_fTSSd[j]=np.sum(fTSSmask)
        fTSSd[j]=np.sum(fTSSl[j::48][fTSSmask])
        
        #  TSS Diurnal
        TSSmask=(TSSl[j::48]>0)&(TSSl[j::48]<1.5)
        c_TSSd[j]=np.sum(TSSmask)
        TSSd[j]=np.sum(TSSl[j::48][TSSmask])
        
    time=np.linspace(0,23.5,48)
    ax=plt.subplot(3,3,i+1)
    print(utcoff)
    l1=ax.plot(time,shift(TSSd/c_TSSd,utcoff*2),'--',c='red')
    l2=ax.plot(time,shift(fTSSd/c_fTSSd,utcoff*2),'--',c='darkorange',linewidth=2)
    ax2=ax.twinx()
    l3=ax2.plot(time,shift(Hd/c_Hd,utcoff*2),c='black')
    plt.title(site_name,fontsize=15)
    
    if i in [0,1,3,4,6,7]:
        ax2.axes.yaxis.set_visible(False)
    if i in [0,3,6]:
        ax.set_ylabel('Temperature Variance $\sigma_T$',color='red',fontsize=12)
    if i in [1,2,4,5,7,8]:
        ax.axes.yaxis.set_visible(False)
    if i in [2,5,8]:
        ax2.set_ylabel('Sensible Heat ($W/m^2$)',color='black',fontsize=12)
    if i in [6,7,8]:
        ax.set_xlabel('Hour of the Day')
    if i <6:
        ax.axes.xaxis.set_visible(False)
    if i == 2:
        plt.legend([l1[0],l2[0],l3[0]],['Unfiltered $\sigma_T$','Filtered $\sigma_T$','Sensible Heat'])
    fpl.close()
    ax.set_ylim(.1,.8)
    ax.set_xlim(6.75,17.25)
    ax2.set_ylim(-50,250)
    i=i+1
plt.subplots_adjust(hspace=.2,wspace=.05)
#fig.suptitle('Diurnal Cycles of Sensible Heat, $\sigma_{T}$ filtered and unfiltered',fontsize=20)

# %%
# --------------------------------------- #
# Figure 3. ALT Diurnal Cycles but Better #
# --------------------------------------- #
#### FUNCTION #####
def shift(arr,off):
    n=arr.shape[0]
    out=np.zeros((n,))
    k=int(round(off))
    for l in range(n):
        if k>=n:
            k=0
        out[k]=arr[l]
        k=k+1
    return out
        
fig=plt.figure(figsize=(11,8))
i=0
for site in focus_sites:
    if site=='MAP':
        ax= plt.subplot(3,3,i+1,projection=cartopy.crs.PlateCarree())
        ax.background_img('ne_shaded','low')
        ax.set_xlim(np.min(nlon)-3,np.max(nlon)+1)
        ax.set_ylim(np.min(nlat)-1,np.max(nlat)+1)
        ax.add_feature(cartopy.feature.COASTLINE)
        #ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.BORDERS)
        #ax.add_feature(cartopy.feature.LAKES)
        ax.set_aspect('auto')
        plt.scatter(nlon,nlat,c='g',s=50,alpha=.5)
        plt.scatter(fnlon,fnlat,marker='*',c='y',s=300)
        ax.axis('off')
        ax.outline_patch.set_edgecolor('white')
        i=i+1
        continue
    site_name=str(site)[2:-1]
    
    # FIXME HERE
    m_site = site==fpsites
    Hs=fp['H'][m_site]
    TSs=fp['T_SONIC_SIGMA'][m_site]
    TFs=fp['TSS_FILTER'][m_site]
    ts=fp['TIME'][m_site]
    TSsb=np.zeros((24,))
    bincount=np.ones((24,))*.01
    TFsb=np.zeros((24,))
    Hsb=np.zeros((24,))
    
    for j in range(len(ts)):
        t = ts[j]
        dt=datetime.datetime.utcfromtimestamp(t)
        h=dt.hour
        TSsb[h]=TSsb[h]+TSs[j]
        TFsb[h]=TFsb[h]+TFs[j]
        Hsb[h]=Hsb[h]+Hs[j]
        bincount[h]=bincount[h]+1
    if site==b'ORNL':
        bincount[0]=.01
        
    time=np.linspace(.5,23.5,24)
    time=time.astype(int)
    ax=plt.subplot(3,3,i+1)
    utcoff=fp['utc_off'][m_site][0]
    TSplt=TSsb/bincount
    TSplt[bincount<1]=float('nan')
    TFplt=TFsb/bincount
    TFplt[bincount<1]=float('nan')
    Hplt=Hsb/bincount
    Hplt[bincount<1]=float('nan')
    
    l1=ax.plot(time,shift(TSplt,utcoff),'--',c='red')
    l2=ax.plot(time,shift(TFplt,utcoff),'--',c='darkorange',linewidth=2)
    ax2=ax.twinx()
    l3=ax2.plot(time,shift(Hplt,utcoff),c='black')
    plt.title(site_name,fontsize=15)
    
    if i in [0,1,3,4,6,7]:
        #ax2.axes.yaxis.set_visible(False)
        ax2.set_yticks([-25,25,75,125,175,225,275,325])
        ax2.axes.yaxis.set_ticklabels([])
        for tic in ax2.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        pass
    if i in [0,3,6]:
        ax.set_ylabel('$\sigma_T$',color='red',fontsize=12)
        ax.set_yticks([.1,.2,.3,.4,.5,.6,.7,.8])
    if i in [1,2,4,5,7,8]:
        #ax.axes.yaxis.set_visible(False)
        ax.set_yticks([.1,.2,.3,.4,.5,.6,.7,.8])
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        ax.axes.yaxis.set_ticklabels([])
        pass
    if i in [2,5,8]:
        ax2.set_ylabel('H ($W/m^2$)',color='black',fontsize=12)
        ax2.set_yticks([-25,25,75,125,175,225,275,325])
    if i in [6,7,8]:
        ax.set_xlabel('Hour of the Day')
        ax.set_xticks([6,8,10,12,14,16,18])
        ax.set_xticklabels(ax.get_xticks(), rotation = 35)
    if i <6:
        ax.set_xticks([6,8,10,12,14,16,18])
        ax.axes.xaxis.set_ticklabels([])
        #ax.axes.xaxis.set_visible(False)
        pass
    if i == 2:
        plt.legend([l1[0],l2[0],l3[0]],['Unfiltered $\sigma_T$','Filtered $\sigma_T$','Sensible Heat'],loc='upper right',fontsize='x-small')
    fpl.close()
    ax.set_ylim(.05,.85)
    ax.set_xlim(5.75,18.25)
    ax2.set_ylim(-50,350)
    i=i+1
plt.subplots_adjust(hspace=.2,wspace=.1)
#fig.suptitle('Diurnal Cycles of Sensible Heat, $\sigma_{T}$ filtered and unfiltered',fontsize=20)

# %%
# ------------------------ #
# Figure 4. Stability #
# ------------------------ #
plt.figure(figsize=(12,4))
#zL = fp['ZL'][:]
zL=fp['ZL2'][:]
ustar=fp['USTAR'][:]
rho=fp['RHO'][:]
H=fp['H'][:]/1005/rho

m = zL<0
plt.subplot(1,2,1)
plt.hexbin(zL[m],np.sqrt(ustar**2/H**2*result**2)[m],gridsize=300,cmap='terrain',mincnt=1,extent=(-2.5,0,0.25,3))
zL2=np.linspace(-2.5,.1,100)
a = 4
b = 8.3
andre_perfect=a*(1-b*zL2)**(-.66)
andre_perfect[zL2>0]=a
yyy=(-zL2)**(-1/3)*.95
plt.plot(zL2,np.sqrt(andre_perfect),'k-',linewidth=2)
plt.plot(zL2,yyy,'r-',linewidth=2)
plt.ylim(.25,3)
plt.xlim(-2.5,.1)
plt.xlabel('$\zeta$',fontsize=14)
plt.ylabel('${\sigma_T}/{T_*}$',fontsize=14)
plt.legend(['EQ 1','EQ 2'],fontsize=14,loc='upper left')
#plt.title('Monin-Obukhov Stability vs Heat Term',fontsize=18)

# ------------------------ #
# Figure 4. Stability 1/3  #
# ------------------------ #
#zL = fp['ZL'][:]
zL=fp['ZL2'][:]
ustar=fp['USTAR'][:]
rho=fp['RHO'][:]
H=fp['H'][:]/1005/rho
yy=np.sqrt(ustar**2/H**2*result**2)
xx=(-zL)**(-1/3)
plt.subplot(1,2,2)
plt.hexbin(xx[m],yy[m],gridsize=300,cmap='terrain',mincnt=1,extent=(0,5,0,5))
xxx=np.linspace(0,5)
xxx2=-xxx**(-3)
yyy=np.sqrt(4*(1-8.3*xxx2)**(-2/3))
plt.plot(xxx,yyy,'k-')
plt.plot([0,5],[0,5*.95],'r-',linewidth=2)
plt.xlabel('$(-\zeta)^{-1/3}$',fontsize=14)
plt.ylabel('${\sigma_T}/{T_*}$',fontsize=14)
plt.xlim(0,5)
plt.ylim(0,5)
plt.legend(['EQ 1','EQ 2'],fontsize=14,loc='upper left')
#plt.title('Monin-Obukhov Stability vs Heat Term',fontsize=18)

# %%
# ------------------------ #
# Figure 4. Stability 1/3  #
# ------------------------ #
plt.figure(figsize=(7,5))
#zL = fp['ZL'][:]
zL=fp['ZL2'][:]
ustar=fp['USTAR'][:]
rho=fp['RHO'][:]
H=fp['H'][:]/1005/rho
yy=np.sqrt(ustar**2/H**2*result**2)
xx=(-zL)**(-1/3)
plt.hexbin(xx,yy,gridsize=300,cmap='terrain',mincnt=1,extent=(0,5,0,5))
xxx=np.linspace(0,5)
xxx2=-xxx**(-3)
yyy=np.sqrt(4*(1-8.3*xxx2)**(-2/3))
plt.plot(xxx,yyy,'k-')
plt.plot([0,5],[0,5*.95],'r-',linewidth=2)
plt.xlabel('$(-\zeta)^{-1/3}$',fontsize=14)
plt.ylabel('${\sigma_T}/{T_*}$',fontsize=14)
plt.xlim(0,5)
plt.ylim(0,5)
plt.legend(['EQ 1','EQ 2'],fontsize=14,loc='upper left')
#plt.title('Monin-Obukhov Stability vs Heat Term',fontsize=18)

# %%
# ------------------------ #
# Figure 5. RF Features Im #
# ------------------------ #
# SETUP
pred = ['H','ZL2','USTAR','WS','CD','T_SONIC','LAI','BLH','CAPE','TCC','TREE','BARE_SOIL','VEG']
n = len(fp['H'][:])
n2 = len(pred)
data = np.zeros((n2,n))
i = 0
for x in pred:
    data[i,:]=fp[x][:]
    if x == 'LAI':
        data[i,:]=data[i,:]/10
    i=i+1
data=np.transpose(data)
# Generate weights
numpoints = {}
for i in range(len(fpsites)):
    site = fpsites[i]
    if site not in numpoints.keys():
        numpoints[site] = 1
    else:
        numpoints[site] = numpoints[site]+1
n = len(sites)
for e in numpoints.keys():
    numpoints[e]=1/numpoints[e]
weights = []
for i in range(len(fpsites)):
    site = fpsites[i]
    weights.append(numpoints[site])

# SPLIT
X_train, X_test, y_train, y_test, w_train,w_test = train_test_split(data,result,weights, test_size=0.2, random_state=0)

# Run RF
clf = sklearn.ensemble.RandomForestRegressor(random_state=1,n_estimators=100,oob_score=True)
clf.fit(X_train,y_train,w_train)

# %%
# ------------------------ #
# Figure 5. RF Features Im #
# ------------------------ #
# PLOTTING 
y_pred = clf.predict(X_test)
plt.figure()
colors_pred={'H':'tomato','USTAR':'aqua','WS':'skyblue','H2O':'slateblue',
             'T_SONIC':'indianred','LAI':'forestgreen','ZL2':'blueviolet',
             'BLH':'gold','TCC':'plum','CAPE':'orange','CD':'peru',
             'VEG':'yellowgreen','TREE':'darkolivegreen','BARE_SOIL':'goldenrod'}
names = {'H':'H',
         'LE':'LE',
         'VPD':'VPD',
         'USTAR':'$u_*$',
         'T_SONIC':'$T_{air}$',
         'WS':'$u\bar$',
         'BLH':'BLH',
         'VEG':'$f_{veg}$',
         'LE':'LE',
         'TREE':'$f_{tree}$',
         'BARE_SOIL':'$f_{bare}$',
         'ZL2':'$\zeta$',
         'LAI':'LAI',
         'TCC':'$f_{cloud}$',
         'CAPE':'CAPE',
         'tow_height':'$h_{tower}$',
         'canopy_height':'$h_{canopy}$',
         'PA':'P',
         'H2O':'$C_{H_{2}O}$',
         'SWC':'SWC',
         'CD':'$C_d$'}


#plt.scatter(y_test,y_pred,alpha=.01)
plt.hexbin(y_test,y_pred,cmap='terrain',gridsize=200,mincnt=1)
limmin = np.min([np.min(y_test),np.min(y_pred)])
limmax = np.max([np.max(y_test),np.max(y_pred)])
plt.plot([limmin,limmax],[limmin,limmax],"k:")
plt.title("RF Validation -- Filtered")
plt.xlabel("Observed Value Sigma_T (K)")
plt.ylabel("Predicted Value Sigma_T (K)")

 #make a names list
namelist=[]
colorlist=[]
predlist=[]
for x in pred:
    namelist.append(names[x])
    colorlist.append(colors_pred[x])
    predlist.append(x)
# sort the feature importance
fIMP = clf.feature_importances_
points = fIMP
fIMP_names={}
fIMP_colors={}
fIMP_pred={}
for i in range(len(namelist)):
    fIMP_names[points[i]]=namelist[i]
for i in range(len(colorlist)):
    fIMP_colors[points[i]]=colorlist[i]
for i in range(len(colorlist)):
    fIMP_pred[points[i]]=predlist[i]
points[::-1].sort()
names2=[]
colors2=[]
pred2=[]
for i in range(len(namelist)):
    colors2.append(fIMP_colors[points[i]])
    names2.append(fIMP_names[points[i]])
    pred2.append(fIMP_pred[points[i]])
    
# plot
plt.figure(figsize=(6,4))
plt.bar(names2,points,color=colors2)
#plt.title("RF Feature Importance")
#plt.xlabel("Predictor")
plt.ylabel("Overall Feature Importance")
plt.xticks(rotation=45,fontsize=12)
plt.grid(False,axis='x')
bias = np.sum(y_pred-y_test)/len(y_pred)
print('Bias: '+str(bias/np.mean(y_test)))
print('Normalized RMSE: '+str(metrics.mean_squared_error(y_test, y_pred,squared=False)/np.mean(y_test)))
print('OOB Score: '+ str(clf.oob_score_))

# %%

# %%
# ---------------------------- #
# Figure 6. SBS RF Features Im #
# ---------------------------- #
### SETUP NOT PLOTTing
weights=np.array(weights)
error={}
bias={}
fimp={}
fimp_unsorted={}
names3={}
pred_order={}
colors_rf={}
#pred = ['H','USTAR','WS','H2O','T_SONIC','LAI','ZL','BLH','TCC','CAPE','SWC']
    
for site in sites:
    if site=='MAP':
        continue
    print(site)
    m_site=fpsites==site
    data2=np.zeros((np.sum(m_site),data.shape[1]))
    for i in range(data.shape[1]):
        data2[:,i]=data[m_site,i]
    X_train2, X_test2, y_train2, y_test2, w_train2,w_test2 = train_test_split(data2,result[m_site],weights[m_site], test_size=0.2, random_state=0)
    clf = sklearn.ensemble.RandomForestRegressor(random_state=1,n_estimators=100,oob_score=True)
    clf.fit(X_train2,y_train2,w_train2)
    y_pred2 = clf.predict(X_test2)
    error[site] = metrics.mean_squared_error(y_test2, y_pred2,squared=False)/np.mean(y_test2)
    bias[site] = np.sum(y_pred2-y_test2)/len(y_pred2)/np.mean(y_test2)
    #make a names list
    namelist=[]
    colorlist=[]
    predlist=[]
    for x in pred:
        namelist.append(names[x])
        colorlist.append(colors_pred[x])
        predlist.append(x)
    # sort the feature importance
    fIMP = clf.feature_importances_
    points = fIMP
    fimp_unsorted[site]=fIMP.copy()
    fIMP_names={}
    fIMP_colors={}
    fIMP_pred={}
    for i in range(len(namelist)):
        fIMP_names[points[i]]=namelist[i]
    for i in range(len(colorlist)):
        fIMP_colors[points[i]]=colorlist[i]
    for i in range(len(colorlist)):
        fIMP_pred[points[i]]=predlist[i]
    points[::-1].sort()
    names2=[]
    colors2=[]
    pred2=[]
    for i in range(len(namelist)):
        colors2.append(fIMP_colors[points[i]])
        names2.append(fIMP_names[points[i]])
        pred2.append(fIMP_pred[points[i]])
    names3[site]=names2
    fimp[site]=points
    colors_rf[site]=colors2
    pred_order[site]=pred2

# %%
# ---------------------------- #
# Figure 6. SBS RF Features Im #
# ---------------------------- #
# Site By Site Feature Importance Plot
shortnames={'H':'H','ZL2':'ZL','WS':'WS','BLH':'BLH','SWC':'SWC','T_SONIC':'T','TCC':'$f_{cloud}$',
            'BARE_SOIL':'$f_bare$','CD':'$C_d$',
            'H2O':'H2O','LAI':'LAI','USTAR':'$U_*$','VEG':'$f_{veg}$','CAPE':'CAPE','TREE':'$f_{tree}$'}
counts={}
shortname_list=[]
longname_list=[]
count_colors=[]
for x in pred:
    counts[x]=0
    count_colors.append(colors_pred[x])
    try:
        shortname_list.append(shortnames[x])
    except:
        continue
    longname_list.append(names[x])


# %%

# %%
# ----------------------------------- #
# Figure 6 (ALT2). SBS RF Features Im #
# ----------------------------------- #
plot_values=np.zeros((len(pred),len(sites)))
for i in range(len(pred)):
    for j in range(len(sites)):
        plot_values[i,j]=fimp_unsorted[sites[j]][i]
plt.figure(figsize=(6,4))
ax=sns.violinplot(data=plot_values.transpose(),palette=count_colors,scale='width',linewidth=.75)
plt.ylim(0,1)
ax.set_xticklabels(longname_list)
plt.xticks(rotation=45,fontsize=12)
plt.ylabel('Site Level Feature Importance')
#plt.title('Distribution of Site Level Feature Importance')

# %%

# %%

# %%
# ----------------------------------- #
# Figure 7a Global fit of a,b         #
# ----------------------------------- #
### FIT BY ITERATION ####

import warnings; warnings.simplefilter('ignore')
#zL = fp['ZL'][:]
zL=fp['ZL2'][:]
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
m = (np.abs(fp['H'][:])>0)&(zL<0)
a_vals = np.linspace(1,20,100)
b_vals = np.linspace(5,40,100)
H = fp['H'][:]/1005/rho
o_Tsig = result.copy()[m]
best={'nRMSE':1,'nBias':1,'a':0,'b':0}
for a in a_vals:
    for b in b_vals:
        andre_RHS = a*(1-b*zL)**(-.66)
        andre_RHS[zL>=0]=a
        andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
        bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)
        rmse2 = metrics.mean_squared_error(o_Tsig, andre_Tsig,squared=False)/np.mean(o_Tsig)
        if rmse2 < best['nRMSE']:
            best['nRMSE']=rmse2
            best['nBias']=bias_2
            best['a']=a
            best['b']=b

# %%
# ----------------------------------- #
# Figure 7a Global fit of a,b PLOT    #
# ----------------------------------- #
# plot fit by iteration
andre_RHS = best['a']*(1-best['b']*zL)**(-.66)
andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
plt.figure(figsize=(5,4))
plt.hexbin(o_Tsig,andre_Tsig,gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.25,0,1.25])
#plt.scatter(o_Tsig,andre_Tsig)
plt.plot([0,1.25],[0,1.25],"k--",linewidth=1)
#plt.title("Equation 1 Fit (a,b) ")
plt.xlabel("Observed $\sigma_T$ (K)")
plt.ylabel("EQ 1 Predicted $\sigma_T$ (K)")
bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)/np.mean(o_Tsig)
print('Bias: '+str(best['nBias']/np.mean(o_Tsig)))
print('nRMSE: '+str(best['nRMSE']))
print('a: '+str(best['a']))
print('b: '+str(best['b']))

# %%
# ----------------------------------- #
# Figure 7b Global fit of C1          #
# ----------------------------------- #
### FIT BY ITERATION ####


import warnings; warnings.simplefilter('ignore')
#zL = fp['ZL'][:]
zL=fp['ZL2'][:]
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
m = (np.abs(fp['H'][:])>0)&(zL<0)
c_vals = np.linspace(.2,2,201)
H = fp['H'][:]/1005/rho
o_Tsig = result.copy()[m]
a=6
best={'nRMSE':1,'nBias':1,'c':0}
for c in c_vals:
    andre_RHS = c**2*(-zL)**(-.66)
    andre_RHS[zL>=0]=a
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
    bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)
    rmse2 = metrics.mean_squared_error(o_Tsig, andre_Tsig,squared=False)/np.mean(o_Tsig)
    if rmse2 < best['nRMSE']:
        best['nRMSE']=rmse2
        best['nBias']=bias_2
        best['c']=c

# %%
# ----------------------------------- #
# Figure 7b Global fit of C1 -- SPLIT #
# ----------------------------------- #
### FIT BY ITERATION ####


import warnings; warnings.simplefilter('ignore')
#zL = fp['ZL'][:]
zL=fp['ZL2'][:]
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
m = (np.abs(fp['H'][:])>0)&(zL<0)
c_vals = np.linspace(.2,2,201)
H = fp['H'][:]/1005/rho
o_Tsig = result.copy()[m]
a=6
best={'nRMSE':1,'nBias':1,'c':0,'tBias':0,'tRMSE':0}
for c in c_vals:
    andre_RHS = c**2*(-zL)**(-.66)
    andre_RHS[zL>=0]=a
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
    a_train, a_test, o_train, o_test = train_test_split(andre_Tsig,o_Tsig, test_size=0.2, random_state=0)
    bias_2 = np.sum(a_train-o_train)/len(o_train)
    rmse2 = metrics.mean_squared_error(o_train, a_train,squared=False)/np.mean(o_train)
    if rmse2 < best['nRMSE']:
        best['nRMSE']=rmse2
        best['nBias']=bias_2
        best['c']=c
        best['tBias'] = np.sum(a_test-o_test)/len(o_test)
        best['tRMSE'] = metrics.mean_squared_error(o_test, a_test,squared=False)/np.mean(o_test)

# %%
print(best)

# %%
# ----------------------------------- #
# Figure 7b Global fit of C1 Plot     #
# ----------------------------------- #
# plot fit by iteration
andre_RHS = best['c']**2*(-zL)**(-.66)
#andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
plt.figure(figsize=(5,4))
plt.hexbin(o_Tsig,andre_Tsig,gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.25,0,1.25])
#plt.scatter(o_Tsig,andre_Tsig)
plt.plot([0,1.25],[0,1.25],"k--",linewidth=1)
#plt.title("Equation 1 Fit (C1) ")
plt.xlabel("Observed $\sigma_T$ (K)")
plt.ylabel("EQ 2 Predicted $\sigma_T$ (K)")
bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)/np.mean(o_Tsig)
print('Bias: '+str(best['nBias']/np.mean(o_Tsig)))
print('nRMSE: '+str(best['nRMSE']))
print('c: '+str(best['c']))

# %%
# ----------------------------------- #
# Figure 7c Global fit of C1 and a,b  #
# ----------------------------------- #
plt.figure(figsize=(6,8))
zL_x=np.log(-zL)
y_ =ustar**2/H**2*result**2
m_nan=(~np.isnan(zL_x))&(~np.isnan(y_))
plt.hexbin(-zL[m_nan],y_[m_nan],xscale='log',gridsize=200,cmap='terrain',mincnt=1,extent=(-2.5,1,0,10))
zL2=-np.logspace(-2.5,1)
a = 4
b = 8.3
andre_perfect_2=a*(1-b*zL2)**(-.66)
a = 7.5
b = 33.9
andre_perfect=a*(1-b*zL2)**(-.66)
andre_perfect[zL2>0]=a
plt.plot(-zL2,andre_perfect_2,'r-')
c1_95=.95**2*(-zL2)**(-.66)
c1_81=.812**2*(-zL2)**(-.66)
rel=1/(1.75+2*(-zL2)**(.66))
plt.plot(-zL2,andre_perfect,'r--')
plt.plot(-zL2,c1_95,'k-')
plt.plot(-zL2,c1_81,'k--')
plt.plot(-zL2,rel,'b:',linewidth=3)
plt.xscale('log')
plt.xlabel('$-\zeta$',fontsize=12)
plt.ylabel('${\sigma_T}^2/{T_*}^2$',fontsize=12)
plt.legend(['EQ1','Fitted EQ1','EQ2','Fitted EQ2','Realizability'],fontsize=12,loc='upper right')
#plt.title('Stability')
#plt.xlim(.001,10)
plt.ylim(0,10)

# %%

# %%

# %%

# %%
# ------------------------------------------ #
# Figure 8 (ALT) 3x3 Fit Global Site by Site #
# ------------------------------------------ #
c=.81 #(.69 or 1)
delta =0
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
zL=fp['ZL2'][:]
andre_RHS = c**2*(-zL)**(-.66)
andre_RHS[zL>=0]=4
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))
o_Tsig = result.copy()

errors={}
tilts={}
fig=plt.figure(figsize=(11,8))
i=0
for site in focus_sites:
    if site=='MAP':
        ax= plt.subplot(3,3,i+1,projection=cartopy.crs.PlateCarree())
        ax.background_img('ne_shaded','low')
        ax.set_xlim(np.min(nlon)-3,np.max(nlon)+1)
        ax.set_ylim(np.min(nlat)-1,np.max(nlat)+1)
        ax.add_feature(cartopy.feature.COASTLINE)
        #ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.BORDERS)
        #ax.add_feature(cartopy.feature.LAKES)
        ax.set_aspect('auto')
        plt.scatter(nlon,nlat,c='g',s=50,alpha=.5)
        plt.scatter(fnlon,fnlat,marker='*',c='y',s=300)
        ax.axis('off')
        ax.outline_patch.set_edgecolor('white')
        i=i+1
        continue
    
    # THE PLOTS
    m_site = fpsites[:]==site
    ax=plt.subplot(3,3,i+1)
    m = np.ones((len(o_Tsig),),dtype=bool)
    m=m_site
    
    zL_x=np.log(-zL)
    y_ =ustar**2/H**2*result**2
    m_nan=(~np.isnan(zL_x))&(~np.isnan(y_))&(m)
    #plt.hexbin(-zL[m_nan],y_[m_nan],xscale='log',gridsize=30,cmap='terrain',mincnt=1,extent=(-2.5,1,0,6))
    plt.scatter(-zL[m_nan],y_[m_nan],alpha=.25,s=5,c='g')
    zL2=-np.logspace(-2.5,1)
    a = 7.5
    b = 33.9
    andre_perfect=a*(1-b*zL2)**(-.66)
    andre_perfect[zL2>0]=a
    plt.plot(-zL2,andre_perfect,'r--')
    a = 4
    b = 8.3
    andre_perfect_2=a*(1-b*zL2)**(-.66)
    c1_95=.95**2*(-zL2)**(-.66)
    c1_81=.812**2*(-zL2)**(-.66)
    plt.plot(-zL2,andre_perfect_2,'r-')
    #plt.plot(-zL2,c1_95,'-',c='black')
    #plt.plot(-zL2,c1_81,'--',c='black')
    
    plt.xscale('log')
    #plt.xlabel('-zL',fontsize=12)
    #plt.ylabel('${\sigma_T}^2/{T_*}^2$',fontsize=12)
    plt.title(str(site)[2:-1],fontsize=15)
    plt.xlim(.01,5)
    plt.ylim(0,6)
    
    if i in [0,1,2,3,4,5]:
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        ax.axes.xaxis.set_ticklabels([])
    if i in [0,3,6]:
        ax.set_ylabel('${\sigma_T}^2/{T_*}^2$',fontsize=12)
    if i in [1,2,4,5,7,8]:
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        ax.axes.yaxis.set_ticklabels([])
    if i in [6,7,8]:
        ax.set_xlabel('-$\zeta$',fontsize=12)
    if i == 2:
        plt.legend(['Fitted EQ 1','EQ 1'],loc='upper right')
    i=i+1
plt.subplots_adjust(hspace=.15,wspace=.05)

# %%

# %%
# ------------------- #
# NLCD PREP NO FIGURE #
# ------------------- #
nlcd_dom = fp['nlcd_dom'][:]
sites =[]
fpsites = fp['site'][:]
npoints =0
apoints =0
nsites=0
asites=0
nlcds = {}
site_nlcds={}
nlcd_sums={}
site_nlcd_dom={}
i=0
for point in fpsites:
    if point not in sites:
        sites.append(point)
        site_nlcds[point]=[]
        for k in fp.keys():
            if 'nlcd' in str(k):
                if k=='nlcd_dom':
                    continue
                site_nlcds[point].append(fp[k][i])
                if str(k) not in nlcds.keys():
                    nlcds[str(k)]=[]
                    nlcd_sums[str(k)]=0
                nlcds[str(k)].append(fp[k][i])
                nlcd_sums[str(k)]=nlcd_sums[str(k)]+fp[k][i]
    if 'x-' in str(point):
        npoints=npoints+1
    else:
        apoints=apoints+1
    i = i+1
for site in sites:
    site_nlcd_dom[site]=nlcd_dom[fpsites==site][0]
    if 'x-' in str(site):
        nsites=nsites+1
    else:
        asites=asites+1
for k in nlcd_sums.keys():
    print(k+": "+str(nlcd_sums[k]))
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wetland',95:'Herb Wet'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue'}
nlcd_labels=[]
nlcd_colors=[]
nlcd_tots=np.zeros((len(nlcds['nlcd21'],)),dtype='float')
start = 0

# %%
from sklearn.linear_model import LinearRegression
def tilt(obs,pr):
    model = LinearRegression()
    model.fit(obs.reshape((-1, 1)),pr)
    return model.coef_[0]


# %%
# ------------------------------------ #
# Figure 9 (Original) Slopes Bar Graph #
# ------------------------------------ #
from matplotlib.patches import Patch

a = 4
b = 8.3
delta =0
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
zL=fp['ZL2'][:]
andre_RHS = a*(1-b*zL)**(-.66)
andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))
o_Tsig = fp['TSS_FILTER'][:]

errors={}
biases={}
tower_height={}
tilts={}
canh={}
for site in sites:
    m_site = fpsites[:]==site
    m = np.ones((n,),dtype=bool)
    m=m_site
    errors[site]=metrics.mean_squared_error(o_Tsig[m], andre_Tsig[m],squared=False)/np.mean(o_Tsig)
    biases[site]=np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])/np.mean(o_Tsig[m]-delta)
    tower_height[site]=fp['tow_height'][m][0]
    canh[site]=fp['canopy_height'][m][0]
    tilts[site]=tilt(o_Tsig[m],andre_Tsig[m])
    #tilts[site]=np.mean(fp['ZD_SEASONAL'][m])
    i=i+1
    
dom_color=[]
slpe=[]
nme=[]
errs=[]
towh=[]
plt.figure(figsize=(12,3.5))
ordered_sitelist=[]
flip_slope={}
for k in tilts.keys():
    flip_slope[tilts[k]]=k
slpe_list=list(flip_slope.keys())
slpe_list.sort()
for k in slpe_list:
    ordered_sitelist.append(flip_slope[k])
#plt.plot([-2,40],[1,1],'--',c='black',alpha=.5)
for site in ordered_sitelist:
    dom_color.append(class_colors[site_nlcd_dom[site]])
    slpe.append(tilts[site])
    nme.append(site)
    errs.append(errors[site])
    towh.append(tower_height[site])
bar1 = plt.bar(nme,slpe,color=dom_color)
plt.ylim(.93,1.6)
plt.xticks(rotation=45)
plt.xlim(-1,39)
i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(errs[i]*100))), ha='center', va='bottom',color='r')
    i=i+1
print(site_nlcd_dom.values())
#plt.title('Best Fit Slope by Site with Landcover Type with Error')
plt.ylabel('Best Fit Slope')
legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))
plt.legend(title='Dominant Landcover',handles=legend_elements)

# %%
# ------------------------------------ #
# Figure 9 (Sensible Heat) Slopes Bar Graph #
# ------------------------------------ #
from matplotlib.patches import Patch

a = 4
b = 8.3
delta =0
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
zL=fp['ZL2'][:]
andre_RHS = a*(1-b*zL)**(-.66)
andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))
o_Tsig = fp['TSS_FILTER'][:]

errors={}
biases={}
tower_height={}
tilts={}
canh={}
Bs_ = {}
for site in sites:
    m_site = fpsites[:]==site
    m = np.ones((n,),dtype=bool)
    m=m_site
    errors[site]=metrics.mean_squared_error(o_Tsig[m], andre_Tsig[m],squared=False)/np.mean(o_Tsig)
    biases[site]=np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])/np.mean(o_Tsig[m]-delta)
    tower_height[site]=fp['tow_height'][m][0]
    canh[site]=fp['canopy_height'][m][0]
    tilts[site]=tilt(o_Tsig[m],andre_Tsig[m])
    Bs_[site]=np.nanmedian(bowen[m])
    #tilts[site]=np.mean(fp['ZD_SEASONAL'][m])
    i=i+1
    
slpe=[]
nme=[]
errs=[]
towh=[]
Bsss=[]
fig=plt.figure(figsize=(12,3.5))
ordered_sitelist=[]
flip_slope={}
for k in tilts.keys():
    flip_slope[tilts[k]]=k
slpe_list=list(flip_slope.keys())
slpe_list.sort()
for k in slpe_list:
    ordered_sitelist.append(flip_slope[k])
#plt.plot([-2,40],[1,1],'--',c='black',alpha=.5)
for site in ordered_sitelist:
    dom_color.append(class_colors[site_nlcd_dom[site]])
    slpe.append(tilts[site])
    nme.append(site)
    errs.append(errors[site])
    towh.append(tower_height[site])
    Bsss.append(Bs_[site])
my_cmap = plt.get_cmap("terrain")
vmin=np.min(Bsss)
vmax=np.max(Bsss)
Hs_r =np.array(Bsss)
Hs_r =(Hs_r-vmin)/(vmax-vmin)
bar1 = plt.bar(nme,slpe,color=my_cmap(Hs_r))
plt.ylim(.92,1.55)
plt.xticks(rotation=45)
plt.xlim(-1,39)
i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(errs[i]*100))), ha='center', va='bottom',color='r')
    i=i+1
print(site_nlcd_dom.values())
#plt.title('Best Fit Slope by Site with Landcover Type with Error')
plt.ylabel('Best Fit Slope',fontsize=14)
legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))

norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=norm)
sm.set_array([])

cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
cb = fig.colorbar(sm, cax=cb_ax)
#cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('Bowen Ratio $H/LE$',fontsize=14)
    
#ax = plt.gca()
#cb1 = mpl.colorbar.ColorbarBase(ax, cmap=my_cmap,
#                                norm=norm)
#plt.legend(title='Dominant Landcover',handles=legend_elements)

# %%

# %%
# ------------------------------------ #
# Figure 9 (BOWEN) Slopes Bar Graph #
# ------------------------------------ #
from matplotlib.patches import Patch

a = 4
b = 8.3
delta =0
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
zL=fp['ZL2'][:]
andre_RHS = a*(1-b*zL)**(-.66)
andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))
o_Tsig = fp['TSS_FILTER'][:]

errors={}
biases={}
tower_height={}
tilts={}
canh={}
Hs_ = {}
for site in sites:
    m_site = fpsites[:]==site
    m = np.ones((n,),dtype=bool)
    m=m_site
    errors[site]=metrics.mean_squared_error(o_Tsig[m], andre_Tsig[m],squared=False)/np.mean(o_Tsig)
    biases[site]=np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])/np.mean(o_Tsig[m]-delta)
    tower_height[site]=fp['tow_height'][m][0]
    canh[site]=fp['canopy_height'][m][0]
    tilts[site]=tilt(o_Tsig[m],andre_Tsig[m])
    Hs_[site]=np.mean(fp['H'][m])
    #tilts[site]=np.mean(fp['ZD_SEASONAL'][m])
    i=i+1
    
slpe=[]
nme=[]
errs=[]
towh=[]
Hsss=[]
fig=plt.figure(figsize=(12,3.5))
ordered_sitelist=[]
flip_slope={}
for k in tilts.keys():
    flip_slope[tilts[k]]=k
slpe_list=list(flip_slope.keys())
slpe_list.sort()
for k in slpe_list:
    ordered_sitelist.append(flip_slope[k])
#plt.plot([-2,40],[1,1],'--',c='black',alpha=.5)
for site in ordered_sitelist:
    dom_color.append(class_colors[site_nlcd_dom[site]])
    slpe.append(tilts[site])
    nme.append(site)
    errs.append(errors[site])
    towh.append(tower_height[site])
    Hsss.append(Hs_[site])
my_cmap = plt.get_cmap("coolwarm")
Hs_r =np.array(Hsss)
Hs_r =(Hs_r-np.min(Hs_r))/(np.max(Hs_r)-np.min(Hs_r))
bar1 = plt.bar(nme,slpe,color=my_cmap(Hs_r))
plt.ylim(.92,1.55)
plt.xticks(rotation=45)
plt.xlim(-1,39)
i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(errs[i]*100))), ha='center', va='bottom',color='r')
    i=i+1
print(site_nlcd_dom.values())
#plt.title('Best Fit Slope by Site with Landcover Type with Error')
plt.ylabel('Best Fit Slope',fontsize=14)
legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))

norm = mpl.colors.Normalize(vmin=np.min(Hsss),vmax=np.max(Hsss))
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=norm)
sm.set_array([])

cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
cb = fig.colorbar(sm, cax=cb_ax)
#cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('Sensible Heat ($Wm^{-2}$)',fontsize=14)
    
#ax = plt.gca()
#cb1 = mpl.colorbar.ColorbarBase(ax, cmap=my_cmap,
#                                norm=norm)
#plt.legend(title='Dominant Landcover',handles=legend_elements)

# %%
import matplotlib.cm as cm

# %%
# ------------------------- #
# Figure 10 Local Fit Stuff #
# ------------------------- #
#### STATIC DATA MUST BE REPLACED ######
# ANDREFIT_T_aconst.ipynb
a_s={b'BART': 4, b'SOAP': 4, b'DELA': 4, b'NIWO': 4, b'MLBS': 4, b'OSBS': 4, b'WOOD': 4, b'TREE': 4, b'CLBJ': 4, b'OAES': 4, b'LENO': 4, b'JERC': 4, b'ORNL': 4, b'TALL': 4, b'BLAN': 4, b'SERC': 4, b'SRER': 4, b'SJER': 4, b'KONZ': 4, b'TEAK': 4, b'NOGP': 4, b'RMNP': 4, b'UNDE': 4, b'DCFS': 4, b'WREF': 4, b'JORN': 4, b'UKFS': 4, b'STER': 4, b'STEI': 4, b'GRSM': 4, b'YELL': 4, b'HARV': 4, b'MOAB': 4, b'SCBI': 4, b'ABBY': 4, b'DSNY': 4, b'CPER': 4, b'ONAQ': 4, b'KONA': 4}
b_s={b'BART': 21.15, b'SOAP': 11.649999999999999, b'DELA': 18.299999999999997, b'NIWO': 24.95, b'MLBS': 29.7, b'OSBS': 51.55, b'WOOD': 52.5, b'TREE': 42.05, b'CLBJ': 13.549999999999999, b'OAES': 41.1, b'LENO': 86.7, b'JERC': 44.9, b'ORNL': 41.1, b'TALL': 10.7, b'BLAN': 11.649999999999999, b'SERC': 24.95, b'SRER': 36.349999999999994, b'SJER': 100.0, b'KONZ': 35.4, b'TEAK': 100.0, b'NOGP': 27.799999999999997, b'RMNP': 18.299999999999997, b'UNDE': 36.349999999999994, b'DCFS': 33.5, b'WREF': 45.85, b'JORN': 69.6, b'UKFS': 20.2, b'STER': 52.5, b'STEI': 28.75, b'GRSM': 7.85, b'YELL': 24.95, b'HARV': 26.849999999999998, b'MOAB': 53.449999999999996, b'SCBI': 15.45, b'ABBY': 15.45, b'DSNY': 28.75, b'CPER': 77.2, b'ONAQ': 100.0, b'KONA': 16.4}
# ANDREFIT_T.ipynb
a_s={b'BART': 5.779999999999999, b'SOAP': 3.44, b'DELA': 6.14, b'NIWO': 6.68, b'MLBS': 5.96, b'OSBS': 8.48, b'WOOD': 10.82, b'TREE': 6.859999999999999, b'CLBJ': 4.16, b'OAES': 9.559999999999999, b'LENO': 8.3, b'JERC': 8.120000000000001, b'ORNL': 6.5, b'TALL': 3.26, b'BLAN': 4.52, b'SERC': 5.42, b'SRER': 8.120000000000001, b'SJER': 11.18, b'KONZ': 8.120000000000001, b'TEAK': 14.78, b'NOGP': 8.120000000000001, b'RMNP': 4.52, b'UNDE': 6.68, b'DCFS': 8.120000000000001, b'WREF': 7.3999999999999995, b'JORN': 11.719999999999999, b'UKFS': 5.0600000000000005, b'STER': 11.36, b'STEI': 6.68, b'GRSM': 2.9, b'YELL': 5.42, b'HARV': 5.6, b'MOAB': 10.64, b'SCBI': 4.52, b'ABBY': 5.779999999999999, b'DSNY': 7.58, b'CPER': 14.42, b'ONAQ': 13.879999999999999, b'KONA': 5.6}
b_s={b'BART': 21.15, b'SOAP': 11.649999999999999, b'DELA': 18.299999999999997, b'NIWO': 24.95, b'MLBS': 29.7, b'OSBS': 51.55, b'WOOD': 52.5, b'TREE': 42.05, b'CLBJ': 13.549999999999999, b'OAES': 41.1, b'LENO': 86.7, b'JERC': 44.9, b'ORNL': 41.1, b'TALL': 10.7, b'BLAN': 11.649999999999999, b'SERC': 24.95, b'SRER': 36.349999999999994, b'SJER': 100.0, b'KONZ': 35.4, b'TEAK': 100.0, b'NOGP': 27.799999999999997, b'RMNP': 18.299999999999997, b'UNDE': 36.349999999999994, b'DCFS': 33.5, b'WREF': 45.85, b'JORN': 69.6, b'UKFS': 20.2, b'STER': 52.5, b'STEI': 28.75, b'GRSM': 7.85, b'YELL': 24.95, b'HARV': 26.849999999999998, b'MOAB': 53.449999999999996, b'SCBI': 15.45, b'ABBY': 15.45, b'DSNY': 28.75, b'CPER': 77.2, b'ONAQ': 100.0, b'KONA': 16.4}
nrmse_s={b'BART': 0.10947336128486457, b'SOAP': 0.14833695984883866, b'DELA': 0.19379958607193376, b'NIWO': 0.08371207662594997, b'MLBS': 0.0880609034443107, b'OSBS': 0.08531184943701674, b'WOOD': 0.07495063976574243, b'TREE': 0.10309425076301157, b'CLBJ': 0.08901328567090996, b'OAES': 0.07154120680850562, b'LENO': 0.13990280416169065, b'JERC': 0.11598149069678673, b'ORNL': 0.10129524244283784, b'TALL': 0.14339948996565138, b'BLAN': 0.11119920737108405, b'SERC': 0.1189101720397206, b'SRER': 0.0697862087500288, b'SJER': 0.10725318005605299, b'KONZ': 0.09051866650300378, b'TEAK': 0.11452394376731322, b'NOGP': 0.10226687541752373, b'RMNP': 0.11535054009170741, b'UNDE': 0.11441736666864834, b'DCFS': 0.09224710546193719, b'WREF': 0.09627384876138559, b'JORN': 0.06922730597993744, b'UKFS': 0.09978866790060789, b'STER': 0.08218940937346741, b'STEI': 0.11018533662124196, b'GRSM': 0.21019651203033599, b'YELL': 0.10224505055880939, b'HARV': 0.10268390114163177, b'MOAB': 0.08062257755801139, b'SCBI': 0.135396220101128, b'ABBY': 0.10125308775865012, b'DSNY': 0.07605426948455019, b'CPER': 0.10247958703384252, b'ONAQ': 0.08146503065175488, b'KONA': 0.10821347182170844}
nbias_s={b'BART': -0.014893426317371259, b'SOAP': -0.025342899124281646, b'DELA': -0.033710547610892524, b'NIWO': -0.012465197571608747, b'MLBS': -0.00706700286907491, b'OSBS': -0.008735632935438659, b'WOOD': -0.008606040469893947, b'TREE': -0.012563277166236987, b'CLBJ': -0.008858875774029766, b'OAES': -0.004816087194169465, b'LENO': -0.021359199558192533, b'JERC': -0.012455326434238464, b'ORNL': -0.009237070782644475, b'TALL': -0.022453328051458454, b'BLAN': -0.01773715074397152, b'SERC': -0.013871967981424182, b'SRER': -0.004714770192619097, b'SJER': -0.011670025070250714, b'KONZ': -0.013188931761890477, b'TEAK': -0.010842520691593105, b'NOGP': -0.01448504602225055, b'RMNP': -0.017365490464910077, b'UNDE': -0.015525778991103583, b'DCFS': -0.012365673512730392, b'WREF': -0.008733193121063573, b'JORN': -0.0031303993230416306, b'UKFS': -0.010807918601871256, b'STER': -0.009087929317873155, b'STEI': -0.01600866730369142, b'GRSM': -0.04648562257913635, b'YELL': -0.010888017221842605, b'HARV': -0.01314577777519162, b'MOAB': -0.009006435245835363, b'SCBI': -0.02191787931007534, b'ABBY': -0.011973771544220834, b'DSNY': -0.008087697378468407, b'CPER': -0.012580116185301506, b'ONAQ': -0.009550452199961276, b'KONA': -0.012489578841667667}
##### STATIC DATA END

# %%
# ------------------------------------------------------- #
# Figure 10alt Local Fit Stuff a constant from global fit #
# ------------------------------------------------------- #
### Site by Site Fit by Iteration ###
a_s={}
b_s={}
nrmse_s={}
nbias_s={}
ormse_s={}
obias_s={}
import warnings; warnings.simplefilter('ignore')
zL = fp['ZL2'][:]
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
m_0 = np.abs(fp['H'][:])>0&(zL<0)
a_vals = [7.5]
b_vals = np.linspace(20,100,101)
H = fp['H'][:]/1005/rho
for site in sites:
    print(site,end=' ')
    m=(fpsites==site)&(m_0)
    o_Tsig = result.copy()[m]
    best={'nRMSE':1,'nBias':1,'a':0,'b':0}
    for a in a_vals:
        for b in b_vals:
            andre_RHS = a*(1-b*zL)**(-.66)
            andre_RHS[zL>=0]=a
            andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
            bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)
            rmse2 = metrics.mean_squared_error(o_Tsig, andre_Tsig,squared=False)/np.mean(o_Tsig)
            if rmse2 < best['nRMSE']:
                best['nRMSE']=rmse2
                best['nBias']=bias_2
                best['a']=a
                best['b']=b
    a_s[site]=best['a']
    b_s[site]=best['b']
    nrmse_s[site]=best['nRMSE']
    nbias_s[site]=best['nBias']
    a=4
    b=8.3
    andre_RHS = a*(1-b*zL)**(-.66)
    andre_RHS[zL>=0]=a
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
    obias_s[site] = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)/np.mean(o_Tsig)
    ormse_s[site] = np.sqrt(metrics.mean_squared_error(o_Tsig, andre_Tsig)/np.mean(o_Tsig))


# %%

# %%

# %%
# ------------------------------------------------------- #
# Figure 10alt Local Fit Stuff a constant from global fit #
# ------------------------------------------------------- #
# PLOT
plot_colors=[]
canh={}
canhm={}
a_s2={}
b_s2={}
csub='median_radius500'
#csub='avg_radius500'
sitelist=[]
for k in nrmse_s.keys():
    count=np.sum(fpsites==k)
    if count < 100:
        continue
    plot_colors.append(class_colors[site_nlcd_dom[k]])
    m_site = fpsites[:]==k
    canh[k]=fp['canopy_height'][m_site][0]
    canhm[k]=fp[csub][m_site][0]+1
    #canhm[k]=np.mean(fp['ZD_SEASONAL'][m_site])
    a_s2[k]=a_s[k]
    b_s2[k]=b_s[k]
    sitelist.append(str(k)[2:-1])
canhm[b'BLAN']=canh[b'BLAN']+1
c2=np.sqrt(np.array(list(a_s2.values()))*np.array(list(b_s2.values()))**(-2/3))
canh_xaxis=np.array(list(canhm.values()))
fig,ax=plt.subplots(figsize=(8,5))
plt.scatter(canh_xaxis,c2,c=plot_colors,s=100)
#plt.title('Comparing the ratio between a and b to error and landcover')
'''
obs=np.log10(canh_xaxis)
pr=c2
model = LinearRegression()
model.fit(obs.reshape((-1, 1)),pr)
print(model.score(obs.reshape((-1, 1)),pr))
fake_x=np.linspace(.1,50)
fake_y=model.predict(np.log(fake_x).reshape(-1,1))
plt.plot(fake_x,fake_y,'r--')
plt.xscale('log')
'''
fake_x=np.linspace(.1,55,1000)
fit = np.polyfit(np.log(canh_xaxis),c2,deg=1)
fake_y=np.log(fake_x)*fit[0]+fit[1]
plt.plot(fake_x,fake_y,'r--',alpha=.5)
plt.xscale('log')
plt.xlim(1,55)

canh_all=fp[csub][:]
canh_all[fpsites==b'BLAN']=canh[b'BLAN']
c2_all=np.zeros((len(canh_all),))
for i in range(len(canh_xaxis)):
    m=canh_all==canh_xaxis[i]
    c2_all[m]=c2[i]

#for i, txt in enumerate(sitelist):
#    ax.annotate(txt, (canh_xaxis[i],c2[i]))
    
pred_c2=np.log(canh_xaxis)*fit[0]+fit[1]
#pred_c2=np.log(canh_all)*fit[0]+fit[1]
from sklearn.metrics import r2_score
print(r2_score(c2,pred_c2))
#print(r2_score(c2_all,pred_c2))
plt.xlabel('Canopy Height (m)')
plt.ylabel('C2 = $\sqrt{a/b^{2/3}}$')
plt.ylim(.65,.97)

print(fit)

legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))
plt.legend(title='Dominant Landcover',handles=legend_elements)

# %%
# ------------------------ #
# Figure A zd and Canh     #
# ------------------------ #
plt.figure(figsize=(7,6))
zd_st=[]
canh_st=[]
site_names=[]
for site in sites:
    msite=site==fpsites
    #zd_st.append(fp['zd'][msite][0])
    canh_st.append(fp['canopy_height'][msite][0])
    zd_st.append(np.mean(fp['ZD_SEASONAL'][msite]))
    #canh_st.append(fp['median_radius500'][msite][0])
    site_names.append(site)
zd_st=np.array(zd_st)
canh_st=np.array(canh_st)
plt.scatter(canh_st,zd_st,c=plot_colors,s=100)
plt.plot([0,60],[0,40],'--',c='grey',alpha=.7)
plt.xlim(.1,60)
plt.ylim(.5,60)
plt.xscale('log')
plt.yscale('log')
ax=plt.gca()
plt.grid(which='both')
plt.grid(which='major',linewidth=2)
#ax.tick_params(axis='x', which='minor', bottom=True)
ax.set_xticks([.1,.2,.3,.4,.5,.6,.7,.8,.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60],minor=True)
plt.ylabel('$z_d$ (m)')
plt.xlabel('$h_c$ (m)')

legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))
plt.legend(title='Dominant Landcover',handles=legend_elements)

# %%
# ------------------------------------------------------------------------ #
# Figure 10alt2 Local Fit Stuff a constant from global fit -- but B not C2 #
# ------------------------------------------------------------------------ #
# PLOT
plot_colors=[]
canh={}
canhm={}
a_s2={}
b_s2={}
csub='median_radius500'
#sub='90pct_radius500'
#csub='avg_radius500'
sitelist=[]
for k in nrmse_s.keys():
    count=np.sum(fpsites==k)
    #if count < 100:
    #   continue
    plot_colors.append(class_colors[site_nlcd_dom[k]])
    m_site = fpsites[:]==k
    canh[k]=fp['canopy_height'][m_site][0]
    canhm[k]=canh[k]
    #canhm[k]=(fp['canopy_height'][m_site][0])**2/((fp['tow_height'][m_site][0]-np.mean(fp['ZD_SEASONAL'][m_site]))**2*np.mean(fp['LAI'][m_site]))
    #if (canh[k]==canhm[k]) & (canh[k]>3):
    #    canhm[k]=fp['75pct_radius500'][m_site][0]
    #canhm[k]=np.mean(fp['ZD_SEASONAL'][m_site])
    a_s2[k]=a_s[k]
    b_s2[k]=b_s[k]
    sitelist.append(str(k)[2:-1])
b13=np.array(list(b_s2.values()))**(-1/3)
#canhm[b'STEI']=canh[b'STEI']
c2=np.sqrt(np.array(list(a_s2.values()))*np.array(list(b_s2.values()))**(-2/3))
canh_xaxis=np.array(list(canhm.values()))
fig,ax=plt.subplots(figsize=(6,3.5))
plt.scatter(canh_xaxis,b13,c=plot_colors,s=100)
#plt.title('Comparing the ratio between a and b to error and landcover')
'''
obs=np.log10(canh_xaxis)
pr=c2
model = LinearRegression()
model.fit(obs.reshape((-1, 1)),pr)
print(model.score(obs.reshape((-1, 1)),pr))
fake_x=np.linspace(.1,50)
fake_y=model.predict(np.log(fake_x).reshape(-1,1))
plt.plot(fake_x,fake_y,'r--')
plt.xscale('log')
'''
minx=.1
maxx=55

fake_x=np.linspace(minx,maxx,1000)
fit = np.polyfit(np.log10(canh_xaxis[canh_xaxis<100]),b13[canh_xaxis<100],deg=1)
fake_y=np.log10(fake_x)*fit[0]+fit[1]
plt.plot(fake_x,fake_y,'r--',alpha=.5)
plt.xscale('log')
plt.xlim(minx,maxx)

'''
for i, txt in enumerate(sitelist):
    if(canh_xaxis[i]==list(canh.values())[i]):
        ax.annotate(txt, (canh_xaxis[i],b13[i]))
'''   
   
pred_b13=np.log(canh_xaxis)*fit[0]+fit[1]
#pred_c2=np.log(canh_all)*fit[0]+fit[1]
from sklearn.metrics import r2_score
print(r2_score(b13,pred_b13))
#print(r2_score(c2_all,pred_c2))
plt.xlabel('$h_c  (m)$')
plt.ylabel('$b^{-1/3}$')
plt.ylim(.22,.36)
#plt.xticks([.5,.6,.7,.8,.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50])

print(fit)

#legend_elements=[]
#keep_nlcds=[41,42,43,52,71,81,82,90]
#for x in keep_nlcds:
#    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))
#plt.legend(title='Dominant Landcover',handles=legend_elements)
#plt.figure()
#plt.scatter(list(canhm.values()),list(canh.values()))
#plt.plot([0,55],[0,55])

# %%
np.max(list(canh.values()))

# %%
# ------------------------------ #
# Evaluating Canopy Height Model #
# ------------------------------ #
b2=np.zeros(len(zL))
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
a=7.5
for i in range(len(fpsites)):
    site=fpsites[i]
    b2[i]=(np.log10(canhm[site])*fit[0]+fit[1])**(-3)
    #b2[i]=b_s2[fpsites[i]]
#b2=(np.log10(fp['ZD_SEASONAL'][:])*fit[0]+fit[1])**(-3)
m = (np.abs(fp['H'][:])>0)&(zL<0)
andre_RHS = a*(1-b2*zL)**(-.66)
andre_RHS[zL>=0]=a
o_Tsig=result.copy()
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
plt.figure()
plt.hexbin(o_Tsig[m],andre_Tsig,gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.25,0,1.25])
#plt.scatter(o_Tsig,andre_Tsig)
plt.plot([0,1.25],[0,1.25],"k--",linewidth=3)
plt.title("Equation 1 Validation ")
plt.xlabel("Eddy Flux Tower Observed Value $\sigma_T$ (K)")
plt.ylabel("EQ 1 Predicted Value $\sigma_T$ (K)")
bias_2 = np.sum(andre_Tsig-o_Tsig[m])/len(o_Tsig[m])/np.mean(o_Tsig[m])
print('Bias: '+str(bias_2))
print('nRMSE: '+str(metrics.mean_squared_error(o_Tsig[m], andre_Tsig,squared=False)/np.mean(o_Tsig[m])))
np.mean(b2)

# %%

# %%
# ------------------------------------------- #
# Evaluating Site by Site Fit (a from global) #
# ------------------------------------------- #
b2=np.zeros(len(zL))
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
a2=np.zeros(len(zL))
c_2={}
#b2=[]
i =0
for site in fpsites:
    b2[i]=b_s[site]
    a2[i]=a_s[site]
    i=i+1
m = (np.abs(fp['H'][:])>0)&(zL<0)
andre_RHS = a2*(1-b2*zL)**(-.66)
andre_RHS[zL>=0]=a
o_Tsig=result.copy()
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
plt.figure()
plt.hexbin(o_Tsig[m],andre_Tsig,gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.25,0,1.25])
#plt.scatter(o_Tsig,andre_Tsig)
plt.plot([0,1.25],[0,1.25],"k--",linewidth=3)
plt.title("Equation 1 Validation ")
plt.xlabel("Eddy Flux Tower Observed Value $\sigma_T$ (K)")
plt.ylabel("EQ 1 Predicted Value $\sigma_T$ (K)")
bias_2 = np.sum(andre_Tsig-o_Tsig[m])/len(o_Tsig[m])/np.mean(o_Tsig[m])
print('Bias: '+str(bias_2))
print('nRMSE: '+str(metrics.mean_squared_error(o_Tsig[m], andre_Tsig,squared=False)/np.mean(o_Tsig[m])))
np.mean(b2)

# %%

# %%
# --------------------------------------- #
# Figure TESTST TEST TEST  #
# --------------------------------------- #
# PLOT
plot_colors=[]
canh={}
canhm={}
a_s2={}
b_s2={}

for ki in fp.keys():
    if 'radius' not in ki:
        continue
    csub=ki
    for k in nrmse_s.keys():
        plot_colors.append(class_colors[site_nlcd_dom[k]])
        m_site = fpsites[:]==k
        canh[k]=fp['canopy_height'][m_site][0]
        canhm[k]=fp[csub][m_site][0]
        a_s2[k]=a_s[k]
        b_s2[k]=b_s[k]
    canhm[b'BLAN']=canh[b'BLAN']
    c2=np.sqrt(np.array(list(a_s2.values()))*np.array(list(b_s2.values()))**(-2/3))
    canh_xaxis=np.array(list(canhm.values()))
    #plt.title('Comparing the ratio between a and b to error and landcover')
    '''
    obs=np.log10(canh_xaxis)
    pr=c2
    model = LinearRegression()
    model.fit(obs.reshape((-1, 1)),pr)
    print(model.score(obs.reshape((-1, 1)),pr))
    fake_x=np.linspace(.1,50)
    fake_y=model.predict(np.log(fake_x).reshape(-1,1))
    plt.plot(fake_x,fake_y,'r--')
    plt.xscale('log')
    '''
    fake_x=np.linspace(.1,55,1000)
    try:
        fit = np.polyfit(np.log(canh_xaxis),c2,deg=1)
    except:
        print(csub+' ERROR ERROR')
        print()
        continue
    fake_y=np.log(fake_x)*fit[0]+fit[1]
    

    canh_all=fp[csub][:]
    canh_all[fpsites==b'BLAN']=canh[b'BLAN']
    c2_all=np.zeros((len(canh_all),))
    for i in range(len(canh_xaxis)):
        m=canh_all==canh_xaxis[i]
        c2_all[m]=c2[i]
    pred_c2=np.log(canh_xaxis)*fit[0]+fit[1]
    #pred_c2=np.log(canh_all)*fit[0]+fit[1]
    from sklearn.metrics import r2_score
    print(csub)
    #print(r2_score(c2,pred_c2))
    #print(r2_score(c2_all,pred_c2))
    b2=np.zeros(len(zL))
    ustar = fp['USTAR'][:]
    rho = fp['RHO'][:]
    H = fp['H'][:]/1005/rho
    a=4
    c_2={}
    for site in a_s.keys():
        c_2[site]=np.log10(canhm[site])*fit[0]+fit[1]
    for i in range(len(fpsites)):
        b2[i]=(c_2[fpsites[i]]**2/a)**(-3/2)
        #b2[i]=b_s2[fpsites[i]]
    m = (np.abs(fp['H'][:])>0)&(zL<0)
    andre_RHS = a*(1-b2*zL)**(-.66)
    andre_RHS[zL>=0]=a
    o_Tsig=result.copy()
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
    bias_2 = np.sum(andre_Tsig-o_Tsig[m])/len(o_Tsig[m])/np.mean(o_Tsig[m])
    print('Bias: '+str(bias_2))
    print('nRMSE: '+str(metrics.mean_squared_error(o_Tsig[m], andre_Tsig,squared=False)/np.mean(o_Tsig[m])))
    print()


# %%
for site in sites:
    print(str(site)+': '+str(np.sum(site==fpsites)))

# %%
### REATTEMPT CANOPY STUFF STATIC ###
# ------------------------------------------------------------------------ #
# TEST TEST TEST TEST TEST TEST TEST TEST -------------------------------- #
# ------------------------------------------------------------------------ #
# PLOT
plot_colors=[]

a_s2={}
b_s2={}
var={}
var['canh']={}
var['canhm']={}
var['z_canh']={}
var['z_canhm']={}
var['z_zd']={}
var['z_zds']={}
var['zzds_canhm']={}
var['lai_canh']={}
var['lai_zds']={}
var['z']={}
var['zd']={}
var['zds']={}
var['u*_u']={}
#var['z-zd_hc-zd']={}
#var['z-zds_hc-zds']={}
var['h2_z-zd2lai']={}
var['h2_z-zds2lai']={}
var['hm2_z-zds2lai']={}
var['tree']={}

csub='90pct_radius500'

islog=True
isExp=False
count={}

sitelist=[]
for k in nrmse_s.keys():
    m_site = fpsites[:]==k
    #if fp['canopy_height'][m_site][0] >=3:
    #    continue
    plot_colors.append(class_colors[site_nlcd_dom[k]])
    sitelist.append(str(k)[2:-1])
    count[k]=np.sum(m_site)
    a_s2[k]=a_s[k]
    b_s2[k]=b_s[k]
    var['z'][k]=fp['tow_height'][m_site][0]
    var['zd'][k]=fp['zd'][m_site][0]
    var['zds'][k]=np.mean(fp['ZD_SEASONAL'][m_site])
    var['canh'][k]=fp['canopy_height'][m_site][0]
    var['canhm'][k]=fp[csub][m_site][0]
    var['z_canh'][k]=fp['tow_height'][m_site][0]/fp['canopy_height'][m_site][0]
    var['z_canhm'][k]=fp['tow_height'][m_site][0]/fp[csub][m_site][0]
    var['z_zd'][k]=fp['tow_height'][m_site][0]/fp['zd'][m_site][0]
    var['z_zds'][k]=fp['tow_height'][m_site][0]/np.mean(fp['ZD_SEASONAL'][m_site])
    var['zzds_canhm'][k]=(fp['tow_height'][m_site][0]-np.mean(fp['ZD_SEASONAL'][m_site]))/fp[csub][m_site][0] 
    var['lai_canh'][k]=np.mean(fp['LAI'][m_site])/fp['canopy_height'][m_site][0]/10
    var['lai_zds'][k]=np.mean(fp['LAI'][m_site])/np.mean(fp['ZD_SEASONAL'][m_site])
    var['u*_u'][k]=np.mean(fp['CD'][m_site])
    var['tree'][k]=np.mean(fp['TREE'][m_site])
    #var['z-zds_hc-zds'][k]=(fp['tow_height'][m_site][0]-np.mean(fp['ZD_SEASONAL'][m_site]))/(fp['canopy_height'][m_site][0]-np.mean(fp['ZD_SEASONAL'][m_site]))
    var['h2_z-zds2lai'][k]=(fp['canopy_height'][m_site][0])**2/((fp['tow_height'][m_site][0]-np.mean(fp['ZD_SEASONAL'][m_site]))**2*np.mean(fp['LAI'][m_site]))
    var['hm2_z-zds2lai'][k]=(fp[csub][m_site][0])**2/((fp['tow_height'][m_site][0]-np.mean(fp['ZD_SEASONAL'][m_site]))**2*np.mean(fp['LAI'][m_site]))
    #var['z-zd_hc-zd'][k]=(fp['tow_height'][m_site][0]-fp['zd'][m_site][0])/(fp['median_radius500'][m_site][0]-fp['zd'][m_site][0])
    var['h2_z-zd2lai'][k]=(fp['canopy_height'][m_site][0])**2/((fp['tow_height'][m_site][0]-fp['zd'][m_site][0])**2*np.mean(fp['LAI'][m_site]))
sitelist=np.array(sitelist)    
    
pltclrs={}
b13={}
for k in var:
    pltclrs[k]=np.array(plot_colors)
    b13[k]=np.array(list(b_s2.values()))**(-1/3)
vara={}
fake_x={}
fake_y={}
rscore={}
site_stuff={}
fits={}
xlims={}
for v in var.keys():
    vara[v]=np.array(list(var[v].values()))
    fake_x[v]=np.linspace(np.min(vara[v])*.9,np.max(vara[v][vara[v]<100])*1.05,1000)
    xlims[v]=[np.min(vara[v])*.9,np.max(vara[v][vara[v]<100])*1.05]
    if islog:
        b13[v]=b13[v][vara[v]<100]
        pltclrs[v]=pltclrs[v][vara[v]<100]
        site_stuff[v]=sitelist[vara[v]<100]
        vara[v]=vara[v][vara[v]<100]
        fit=np.polyfit(np.log10(vara[v]),b13[v],deg=1)
        fits[v]=fit
        fake_y[v]=np.log10(fake_x[v])*fit[0]+fit[1]
        pred_b13=np.log10(vara[v])*fit[0]+fit[1]
        rscore[v]=r2_score(b13[v],pred_b13)
    elif not isExp:
        b13[v]=b13[v][vara[v]<100]
        site_stuff[v]=sitelist[vara[v]<100]
        pltclrs[v]=pltclrs[v][vara[v]<100]
        vara[v]=vara[v][vara[v]<100]
        fit=np.polyfit(vara[v],b13[v],deg=1)
        fits[v]=fit
        fake_y[v]=fake_x[v]*fit[0]+fit[1]
        pred_b13=vara[v]*fit[0]+fit[1]
        rscore[v]=r2_score(b13[v],pred_b13)
    else:
        print('hello')

varanames={'canh':'$h_c (m)$',
           'z_zds':'$z/z_d$',
           'u*_u':'$C_D$',
           'lai_canh':'$LAI/h_c (m^{-1})$',
           'h2_z-zds2lai':'$h_c^2/[(z-z_d)^2LAI]$',
           'lai_zds':'$LAI/z_d (m^{-1})$'}

i = 1
fig,ax=plt.subplots(figsize=(8,8))
for v in varanames.keys():
    plt.subplot(3,2,i)
    i=i+1
    plt.scatter(vara[v],b13[v],c=pltclrs[v],s=40)
    plt.plot(fake_x[v],fake_y[v],'r--',alpha=.5)
    plt.ylim(.22,.36)
    plt.xticks([.0001,.001,.01,.1,1,10,100],['$10^{-4}','$10^{-3}','$10^{-2}','$10^{-1}','$10^{-0}','$10^{1}','$10^{2}'])
    plt.xlim(xlims[v][0],xlims[v][1])
    if i in [2,4,6]:
        plt.ylabel('$b^{-1/3}$')
    plt.xlabel(varanames[v])
    #for i, txt in enumerate(site_stuff[v]):
    #    ax.annotate(txt, (vara[v][i],b13[v][i]))
    if islog:
        plt.xscale('log')
    if v=="z_zds":
        plt.xticks([1,10],['$10^{0}$','$10^{1}$'])
    else:
        pass
    print(v)
    print(rscore[v])
    print()
plt.subplots_adjust(hspace=.4,wspace=.3)

# %%
v='z_zds'
plt.scatter(vara[v],b13[v],c=pltclrs[v],s=40)
plt.plot(fake_x[v],fake_y[v],'r--',alpha=.5)
plt.ylim(.22,.36)
plt.xlim(xlims[v][0],xlims[v][1])
plt.xlabel(varanames[v])
plt.xscale('log')
plt.xticks([.0001,.001,.01,.1,1,10,100],['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{-0}$','$10^{1}$','$10^{2}$'])


# %%
# ------------------------------ #
# Evaluating Canopy Height Model #
# ------------------------------ #
b2=np.zeros(len(zL))
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
a=7.5
for vari in var.keys():
    print(vari)
    for i in range(len(fpsites)):
        site=fpsites[i]
        m_site=fpsites[i]==site
        b2[i]=(np.log10(var[vari][site])*fits[vari][0]+fits[vari][1])**(-3)
        #b2[i]=(var[vari][site]*fits[vari][0]+fits[vari][1])**(-3)
    #b2=(np.log10(fp['ZD_SEASONAL'][:])*fit[0]+fit[1])**(-3)
    m = (np.abs(fp['H'][:])>0)&(zL<0)
    andre_RHS = a*(1-b2*zL)**(-.66)
    andre_RHS[zL>=0]=a
    o_Tsig=result.copy()
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
    bias_2 = np.sum(andre_Tsig-o_Tsig[m])/len(o_Tsig[m])/np.mean(o_Tsig[m])
    print('Bias: '+str(bias_2))
    print('nRMSE: '+str(metrics.mean_squared_error(o_Tsig[m], andre_Tsig,squared=False)/np.mean(o_Tsig[m])))
    np.mean(b2)
    print()

# %%
# ------------------------------------------------------------------------ #
# Figure 10alt2 Local Fit Stuff a constant from global fit -- but B not C2 #
# ------------------------------------------------------------------------ #
# PLOT
plot_colors=[]
canh={}
canhm={}
a_s2={}
b_s2={}
csub='median_radius500'
#sub='90pct_radius500'
#csub='avg_radius500'
sitelist=[]
for k in nrmse_s.keys():
    count=np.sum(fpsites==k)
    plot_colors.append(class_colors[site_nlcd_dom[k]])
    m_site = fpsites[:]==k
    canh[k]=fp['canopy_height'][m_site][0]
    canhm[k]=canh[k]
    #canhm[k]=(fp['canopy_height'][m_site][0])**2/((fp['tow_height'][m_site][0]-np.mean(fp['ZD_SEASONAL'][m_site]))**2*np.mean(fp['LAI'][m_site]))
    #if (canh[k]==canhm[k]) & (canh[k]>3):
    #    canhm[k]=fp['75pct_radius500'][m_site][0]
    #canhm[k]=np.mean(fp['ZD_SEASONAL'][m_site])
    a_s2[k]=a_s[k]
    b_s2[k]=b_s[k]
    sitelist.append(str(k)[2:-1])
b13=np.array(list(b_s2.values()))**(-1/3)
#canhm[b'STEI']=canh[b'STEI']
c2=np.sqrt(np.array(list(a_s2.values()))*np.array(list(b_s2.values()))**(-2/3))
canh_xaxis=np.array(list(canhm.values()))
fig,ax=plt.subplots(figsize=(8,5))
plt.scatter(canh_xaxis,b13,c=plot_colors,s=100)
#plt.title('Comparing the ratio between a and b to error and landcover')
plt.xscale('log')
'''
obs=np.log10(canh_xaxis)
pr=c2
model = LinearRegression()
model.fit(obs.reshape((-1, 1)),pr)
print(model.score(obs.reshape((-1, 1)),pr))
fake_x=np.linspace(.1,50)
fake_y=model.predict(np.log(fake_x).reshape(-1,1))
plt.plot(fake_x,fake_y,'r--')
plt.xscale('log')
'''


fake_x=np.linspace(.005,.1,1000)
fit = np.polyfit(np.log10(canh_xaxis),b13,deg=1)
fake_y=np.log10(fake_x)*fit[0]+fit[1]
plt.plot(fake_x,fake_y,'r--',alpha=.5)
plt.xscale('log')
plt.xlim(.005,.1)

#for i, txt in enumerate(sitelist):
#    if(canh_xaxis[i]==list(canh.values())[i]):
#        ax.annotate(txt, (canh_xaxis[i],b13[i]))
    
pred_b13=np.log(canh_xaxis)*fit[0]+fit[1]
#pred_c2=np.log(canh_all)*fit[0]+fit[1]
from sklearn.metrics import r2_score
print(r2_score(b13,pred_b13))
#print(r2_score(c2_all,pred_c2))
plt.xlabel('Canopy Height (m)')
plt.ylabel('$b^{-1/3}$')
#plt.ylim(.22,.36)
plt.xticks([.005,.006,.007,.008,.009,.01,.02,.03,.04,.05,.06,.07,.08,.09,.1])
print(fit)

# %%
# ------------------------------ #
# Evaluating Canopy Height Model #
# ------------------------------ #
b2=np.zeros(len(zL))
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
a=7.5
for i in range(len(fpsites)):
    site=fpsites[i]
    msite=fpsites==site
    #b2[i]=(np.log10(fp['canopy_height'][i]**2/((fp['tow_height'][i]-fp['ZD_SEASONAL'][i])**2*fp['LAI'][i]))*fit[0]+fit[1])**(-3)
    b2[i]=(np.log10(fp['tow_height'][i]/fp['ZD_SEASONAL'][i])*fit[0]+fit[1])**(-3)
#b2=(np.log10(fp['ZD_SEASONAL'][:])*fit[0]+fit[1])**(-3)
m = (np.abs(fp['H'][:])>0)&(zL<0)
andre_RHS = a*(1-b2*zL)**(-.66)
andre_RHS[zL>=0]=a
o_Tsig=result.copy()
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
plt.figure()
plt.hexbin(o_Tsig[m],andre_Tsig,gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.25,0,1.25])
#plt.scatter(o_Tsig,andre_Tsig)
plt.plot([0,1.25],[0,1.25],"k--",linewidth=3)
plt.title("Equation 1 Validation ")
plt.xlabel("Eddy Flux Tower Observed Value $\sigma_T$ (K)")
plt.ylabel("EQ 1 Predicted Value $\sigma_T$ (K)")
bias_2 = np.sum(andre_Tsig-o_Tsig[m])/len(o_Tsig[m])/np.mean(o_Tsig[m])
print('Bias: '+str(bias_2))
print('nRMSE: '+str(metrics.mean_squared_error(o_Tsig[m], andre_Tsig,squared=False)/np.mean(o_Tsig[m])))
np.mean(b2)
print(fit)

# %%
print(count[b'SRER'])

# %%
v='canhm'
fig,ax=plt.subplots(figsize=(8,5))
plt.scatter(vara[v],b13,c=plot_colors,s=100)
plt.plot(fake_x[v],.33*fake_x[v]**-.05)
plt.xscale('log')

# %%
### REATTEMPT CANOPY STUFF CONTINUOUS ###
# ------------------------------------------------------------------------ #
# TEST TEST TEST TEST TEST TEST TEST TEST -------------------------------- #
# ------------------------------------------------------------------------ #
# PLOT
m1=fp['ZL2'][:]<0
rho = fp['RHO'][m1]
Tstar=fp['H'][m1]/1005/rho/fp['USTAR'][m1]
theta=fp['TSS_FILTER'][m1]
a=7.5
b_all=(1-(theta**2/(a*Tstar**2))**(-3/2))/fp['ZL2'][m1]
b_pred=[]
b_static=[]
var=fp['LAI'][m1]/fp['median_radius500'][m1]/10
for i in range(len(b_all)):
    site=fpsites[m1][i]
    b_static.append(b_s[site])
    b_pred.append((np.log10(var[i])*fits['lai_canhm'][0]+fits['lai_canhm'][1])**(-3))

plt.figure()
plt.hexbin(var,b_static,gridsize=200,cmap='terrain',mincnt=1)
plt.title('lai vs b_static')
plt.figure()
plt.hexbin(b_all,b_pred,gridsize=200,cmap='terrain',mincnt=1,extent=(10,50,10,50))
plt.title('ball vs b_pred')
    
    
for site in focus_sites:
    if site=='MAP':
        continue
    m_site=fpsites[m1]==site
    plt.figure()
    print(np.mean(b_all[m_site]))
    plt.hexbin(var[m_site],b_all[m_site],gridsize=200,cmap='terrain',mincnt=1,extent=(0,2,0,200))
    plt.title(site)


# %%

# %%

# %%
xx=np.linspace(.01,50,1000)
yy=xx**-.01
plt.plot(xx,yy)

# %%
fp2=h5py.File('../data/eddy_v2/L2/DELA_L2.h5','r')
plt.plot(fp2['LAI'][:]/10)
print(fp2.keys())
print(fp2.attrs['canopy_height'])

# %%
i=1
plt.subplots(figsize=(16,10))
for site in sites:
    plt.subplot(5,8,i)
    i=i+1
    msite=site==fpsites
    tsig=fp['TSS_FILTER'][msite]
    cd=fp['CD'][msite]
    cdmin=0
    cdmax=.2
    plt.hexbin(cd,tsig,gridsize=50,cmap='terrain',mincnt=1,extent=(0,.2,0,1.25))
    plt.title(site)

# %%
# ----------------------------------- #
# Figure 7a Global fit of a,b PLOT    #
# ----------------------------------- #
# plot fit by iteration
o_Tsig = result.copy()
andre_RHS = best['a']*(1-best['b']*zL)**(-.66)
andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))

# %%
bowen = fp['H'][:]/fp['LE'][:]/1005*2472

# %%
pbp_abserror = np.abs(o_Tsig[:]-andre_Tsig[:])/o_Tsig[:]

# %%
len(bowen)

# %%
site_bowen = []
site_error = []
site_slpes = []
site_H = []
plot_colorss=[]
for site in sites:
    plot_colorss.append(class_colors[site_nlcd_dom[site]])
    site_H.append(np.median(fp['H'][fpsites==site]))
    site_bowen.append(np.median(bowen[fpsites==site]))
    site_error.append(errors[site])
    site_slpes.append(tilts[site])

# %%
plt.figure(figsize=(14,4))
plt.bar(sites,site_bowen)
plt.xticks(rotation=45)
plt.ylim(0,5)

# %%
plt.scatter(site_bowen,site_error,s=75,c=plot_colors)
plt.xscale('log')
plt.xticks([.1,.2,.3,.4,.5,.6,.7,.8,.9,1,10])
plt.ylabel('Site Mean Error in $\sigma_T$ (K)')
plt.xlabel('Site Mean Bowen Ratio')

# %%

# %%

# %%
bowen2 = bowen[:]
bowen2[bowen<=0]=float('NaN')
bowen2[fp['H'][:]<25]=float('NaN')
plt.figure(figsize=(7.5,4.5))
plt.hexbin(bowen2,pbp_abserror*100,gridsize=300,cmap='terrain',xscale='log',mincnt=1, extent=[-1,3,0,80])
plt.xticks([.1,.2,.3,.4,.5,.6,.7,.8,.9,1,2,3,4,5,6,7,8,9,10,20,30,40])
plt.ylim(-2,80)
plt.xlim(.1,20)
plt.xlabel('Bowen Ratio $H/LE$',fontsize=14)
plt.ylabel('Absolute Error in $\sigma_T$ (%)',fontsize=14)

# %%

# %%
i=1
plt.subplots(figsize=(16,10))
for site in sites:
    plt.subplot(5,8,i)
    i=i+1
    bw3 = bowen[fpsites==site]
    plt.hexbin(bw3,pbp_abserror[fpsites==site],gridsize=200,cmap='terrain',mincnt=1, extent=[0,1,0,1])

# %%
bowen2 = bowen[:]
bowen2[bowen<=0]=float('NaN')
bowen2[fp['H'][:]<50]=float('NaN')
plt.figure(figsize=(10,7))
plt.hexbin(bowen2,pbp_abserror,gridsize=100,cmap='terrain',mincnt=1, extent=[0,1.5,0,1])

# %%
plt.scatter(site_H,site_error,s=75,c=plot_colors)
plt.ylabel('Site Mean Error in $\sigma_T$ (K)')
plt.xlabel('Site Mean Sensible Heat ($W m^{-2}$)')

# %%
plt.figure(figsize=(7.5,4.5))
plt.hexbin(fp['H'][:],alt_error,gridsize=300,cmap='terrain',mincnt=1,extent=(0,500,0,.4))

# %%
plt.figure(figsize=(7.5,4.5))
plt.hexbin(fp['H'][:],pbp_abserror*100,gridsize=200,cmap='terrain',mincnt=1,extent=(0,500,0,100))

# %%
alt_error=np.abs(o_Tsig[:]-andre_Tsig[:])

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
