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
#     display_name: Python 3
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

# %%
for site in focus_sites:
    if site=='MAP':
        continue
    fp=h5py.File('/home/tsw35/soteria/data/eddy_v2/old/'+str(site)[2:-1]+'_L2.h5','r')
    print(fp['vertical_wind'].keys())
    print(fp.attrs['tow_height'])
    print()

# %%
diri='/home/tsw35/soteria/neon_code/data/QAQCv5_neon_pntEB0.25_day_rain_frez_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5'

# %%
fp=h5py.File(diri,'r')

# %%
fp.keys()

# %%
msk=fp['ZL'][:]<-.01
print(np.sum(msk))

# %%
ZL=fp['ZL2'][msk]

# %%
result=.4*(fp['tow_height'][msk]+fp['W2_HEIGHT'][msk])/2/fp['USTAR'][msk]
dudz=((fp['WS'][msk]-fp['WIND_2'][msk]))/(fp['tow_height'][msk]-fp['W2_HEIGHT'][msk])
result=result*dudz
msk2=(dudz>0)&(dudz<1)&(fp['USTAR'][msk]<.1)
result=result[msk2]

# %%

# %%
plt.hist(dudz,bins=np.linspace(-.2,.7,50))
plt.show()

# %%
plt.semilogx(-ZL,result,'o',alpha=.01)
#plt.ylim([-1,1.2])
plt.show()

# %%
plt.hexbin(np.log10(-ZL[msk2]),result,gridsize=150,cmap='terrain',mincnt=1,extent=(-2,3,-2,2))
zl_=np.linspace(-100,-.01,1000)
ky=((1+.625*(-zl_)**2)/(1+7.5*(-zl_)))**(1/3)
ky2=(1-16*zl_)**(-1/4)
plt.plot(np.log10(-zl_),ky)
plt.plot(np.log10(-zl_),ky2)

plt.show()

# %%

# %%

# %%
