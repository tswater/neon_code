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
sns.set_theme()


# %%
fp=h5py.File('data/QAQCv4_neon_pntEB5_rain_frez_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5','r')


# %%
zL =fp['ZL2'][:]
plt.figure(figsize=(7,8))
m=(zL>0.0001)#&(np.abs(fp['H'][:])>30)
result=fp['TSS_FILTER'][:]
ustar=fp['USTAR'][:]
rho=fp['RHO'][:]
H=fp['H'][:]/1005/rho
plt.hexbin(zL[m],(ustar**2/H**2*result**2)[m],xscale='log',gridsize=150,cmap='terrain',mincnt=1,extent=(-3,3,0,20))
plt.plot([10**-2.5,10**2],[4,4],'r-')

# %%
xx=(ustar**2/H**2*result**2)[m]

# %%
print(np.mean(xx))

# %%
print(np.median(xx))

# %%

# %%
