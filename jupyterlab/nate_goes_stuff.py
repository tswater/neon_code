import os
import netCDF4 as nc
import numpy as np
import h5py
import datetime
import rasterio

# Create copies of the GOES files, fill them with all the other data
# also add in the masks, one for each setup. 

# ------------------------ #
# QC OPTIONS and CONSTANTS #
# ------------------------ #
goes_dir = '/home/tsw35/soteria/data/eddy_v2/GOES/'
in_dir = '/home/tsw35/soteria/data/eddy_v2/L2/'
goes_out = '/home/tsw35/soteria/proj_monin/data/GOES/'

# run options
opt_night  = True # True exclueds night points
opt_precip = True # True excludes points with precipitation
opt_freeze = True # True exclues points with sub-zero temperatures
opt_EBmin  = 0    # minimum energy balance residual 
opt_EBmid  = .2   # divider to separate 'balanced' and 'unbalanced' EB
opt_EBmax  = 1    # maximum energy balance residual
opt_minrad = 100  # minimum incoming radiation

# core variables
core_var = ['TSS_FILTER','TSS_FILTER_HP','T_SONIC_SIGMA','H','LE','NETRAD',\
            'G','USTAR','T_SONIC','H2O','BLH','VEG','TREE','TCC','CAPE',\
            'ZL','WS','BARE_SOIL','LAI']
# Bad stuff
bad=['LAJA','GUAN']


# --------- #
# CORE CODE #
# --------- #
filelist=[]
for file in os.listdir(in_dir):
    if file[0:4] not in bad:
        filelist.append(file)

for file in filelist:
    print(file)
    # setup stuff
    fp_in = h5py.File(in_dir+file,'r')
    n = len(fp_in['TIME'][:])
    mask = np.ones((n,),dtype=bool)
    utc_off = fp_in.attrs['utc_off']
    utc_dt  = datetime.timedelta(hours=int(utc_off))
    
    # ----------- #
    # SETUP MASKS #
    # ----------- #
    # check basics
    for var in core_var:
        if (var == 'T_SONIC_SIGMA'):
            mask=mask&(fp_in['qT_SONIC'][:]!=1)
        if (var == 'TSS_FILTER'):
            mask=mask&(fp_in[var][:]>=0)
        if (var == 'TSS_FILTER_HP'):
            mask=mask&(~np.isnan(fp_in[var][:])&(fp_in[var][:]>=0))
        mask=mask&(~np.isnan(fp_in[var][:]))
        mask=mask&(fp_in[var][:]>-9999)

    # CHECK PRECIPITATION
    if (opt_precip):
        mask=mask&(fp_in['P'][:]<=.00001)
    
    # CHECK ABOVE FREEZING
    if (opt_freeze):
        try:
            mask=mask&((fp_in['TA'][:]>0)|(fp_in['T_SONIC'][:]>0))
        except:
            mask=mask&(fp_in['T_SONIC'][:]>0)

    # CHECK DAYTIME POINTS
    if (opt_night):
        mask=mask&(fp_in['DAY'][:]==1)

    # ENERGY BALANCE
    ebres=(fp_in['NETRAD'][:]-fp_in['G'][:]-fp_in['H'][:]-fp_in['LE'][:])/\
          ((fp_in['NETRAD'][:]+.00034))
    ebres=np.abs(ebres)
    mask_b=mask&(ebres<opt_EBmid)&(ebres>=opt_EBmin)
    mask_ub=mask&(ebres<opt_EBmax)&(ebres>=opt_EBmid)

    # LOAD IN THE NETCDF
    fp_out=nc.Dataset(goes_out+file[0:4]+'.nc','r+')
    for var in fp_in.keys():
        if var == 'canopy_heights':
            continue
        if var == 'HSS_FILTER':
            continue
        if var == 'vertical_wind':
            continue
        if var == 'SWCs':
            continue
        if var == 'TIME':
            continue
        try: 
            fp_out.createVariable(var,'d',dimensions=('t'))
            fp_out[var][:]=fp_in[var][:]
        except Exception:
            try: 
                fp_out[var][:]=fp_in[var][:]
            except Exception:
                print('ERROR WITH '+var)
        
    # ADD ADITTIONAL VARIABLES
    # add EB res
    try:
        fp_out.createVariable('EB_RES','d',dimensions=('t'))
    except:
        pass
    fp_out['EB_RES'][:]=ebres[:]
    
    # Add Drag
    try:
        fp_out.createVariable('CD','d',dimensions=('t'))
    except:
        pass
    fp_out['CD'][:]=fp_in['USTAR'][:]**2/(fp_in['WS'][:]**2+.001)

    # Mask
    try:
        fp_out.createVariable('MASK_EB','i1',dimensions=('t'))
        fp_out.createVariable('MASK_NOEB','i1',dimensions=('t'))
    except:
        pass
    fp_out['MASK_EB'][:]   = mask_b
    fp_out['MASK_NOEB'][:] = mask_ub
    




