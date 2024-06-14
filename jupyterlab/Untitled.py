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
import h5py
import seaborn as sns
import netCDF4 as nc

# %%
fp=h5py.File('data/noqaqc_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5','r')
res=fp['RES'][:]
nrad=fp['NETRAD'][:]

# %%
sns.set_theme()

# %%
msk=(res<500)&(res>-200)&(nrad>100)
res2=res[msk]
nrad2=nrad[msk]
from scipy import stats

# %%
plt.hist(np.abs(res2),bins=100)
print(np.median(np.abs(res2)))
print(stats.percentileofscore(np.abs(res2)/np.abs(nrad2+.01)*100,10))
plt.figure(figsize=(5,3))
plt.hist(np.abs(res2)/np.abs(nrad2+.01)*100,bins=50,range=(0,125),color='forestgreen')
plt.xlabel('Energy Balance Residual (%)')
plt.xlim(0,125)
plt.ylabel('# of points')

# %%
sites=['HEAL','TOOL','DEJU','BARR','BONA']
dir_='/home/tsw35/soteria/data/eddy_v2/lst'
for site in sites:
    plt.figure()
    fp=nc.Dataset(dir_+'/'+site+'_L2.h5','r')
    plt.plot(fp['LE'][:])
    plt.title(site)
    plt.ylim(-300,300)
plt.show()

# %%

# %%
