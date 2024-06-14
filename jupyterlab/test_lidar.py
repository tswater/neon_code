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
import h5py
import rasterio
import os

# %%

# %%

# %%
checks=['BLAN','STER','STEI','SJER','YELL','OSBS','SOAP','LENO']
files='/home/tsw35/soteria/data/NEON/chm/_tifs/'

# %%
files='/home/tsw35/soteria/data/NEON/chm/_tifs/'
for file in os.listdir(files):
    if file[0:4] not in checks:
        continue
    plt.figure(figsize=(10,10))
    data=rasterio.open(files+file,'r').read(1)
    data[data<0]=float('nan')
    plt.imshow(data)
    plt.title(file)
    plt.colorbar()

# %%
file='ABBY_chm.tif'
plt.figure()
data=rasterio.open(files+file,'r').read(1)
data[data<0]=float('nan')
r=200
data=data[5548-r:5548+r,2749-r:2749+r]
plt.imshow(data,vmin=0,vmax=np.max(data))
plt.colorbar()
import numpy as np
#print(np.mean(data[5548-500:5548+500,2749-500:2749+500]))
plt.scatter(100,100)
plt.title(file)
print(np.max(data))
print(np.median(data))

# %%
