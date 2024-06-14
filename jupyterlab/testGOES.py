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
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# %%
file='../../GOES/ABBY.nc'
fp=nc.Dataset(file,'r')

# %%
plt.figure(figsize=(12,4))
plt.plot(fp['LST'][10000:20000,10,10])

# %%
fp.close()

# %%
