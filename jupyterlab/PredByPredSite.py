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
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.ensemble
from sklearn import metrics

# %%
fp = h5py.File('data/QAQCv4_neon_pntEB0.2_day_rain_frez_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5','r')

# %%
#pred = ['H','USTAR','WS','VEG','TREE','H2O','T_SONIC','BARE_SOIL','LAI','ZL','BLH','TCC','CAPE','SWC']
pred = ['H','USTAR','WS','H2O','T_SONIC','LAI','ZL','BLH','TCC','CAPE','SWC']
#pred = ['H','TREE']
names = {'H':'Sensible Heat Flux',
         'VPD':'Vapor Pressure Deficit',
         'USTAR':'Friction Velocity',
         'T_SONIC':'Temperature',
         'WS':'Windspeed',
         'BLH':'Bndry Layer Ht',
         'VEG':'Low Veg Frac',
         'LE':'Latent Heat Flux',
         'TREE':'Treecover Frac',
         'BARE_SOIL':'Bare Soil Frac',
         'ZL':'MO Stability',
         'LAI':'Leaf Area Index',
         'TCC':'Cloud Cover',
         'CAPE':'CAPE',
         'tow_height':'Tower Height',
         'canopy_height':'Canopy Height',
         'PA':'Pressure',
         'H2O':'Water Vapor',
         'SWC':'Soil Water'}
minmax={'H':[-100,600],'USTAR':[0,1.75],'WS':[0,15],'VEG':[0,100],'TREE':[0,100],'H2O':[0,35],'T_SONIC':[0,35],
        'BARE_SOIL':[0,100],'LAI':[0,70],'ZL':[-1.5,.5],'BLH':[0,4500],'TCC':[0,1],'SWC':[0,.4],'CAPE':[0,10]}
rmin = 0
rmax = 1.275

n = len(fp['H'][:])
result_old=(fp['T_SONIC_SIGMA'][:])
result=(fp['TSS_FILTER'][:])

# %%
# ------------ #
# SITE BY SITE #
# ------------ #
sites =[]
fpsites = fp['site'][:]
for point in fpsites:
    if point not in sites:
        sites.append(point)

# %%
for x in pred:
    i = 1
    fig = plt.figure(figsize=(21,21))
    for site in sites:
        m_site = fpsites[:]==site
        plt.subplot(7,7,i)
        plt.title(str(site)[2:-1])
        plt.hexbin(fp[x][m_site],result[m_site],gridsize=50,cmap='terrain',mincnt=1,extent=[minmax[x][0],minmax[x][1],rmin,rmax])
        i = i+1
    fig.suptitle('Direct Comparison of '+names[x]+' vs $\sigma_T$',fontsize=20)
    plt.subplots_adjust(wspace=.3,hspace=.25)

# %%
