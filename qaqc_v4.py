# Quality Control for NEON and Ameriflux Data

import os
import netCDF4 as nc
import numpy as np
import h5py
import datetime
import ephem
import rasterio

# ----------- #
# USER INPUTS #
# ----------- #

# select variables that each point MUST have valid values for
core_var = ['TSS_FILTER','T_SONIC_SIGMA','H','BLH','VEG','TREE',\
            'H2O','USTAR','T_SONIC','ZL','WS','BARE_SOIL','LAI',\
            'TCC','CAPE','SWC']
#core_var = ['H2OS_FILTER','H2O_SIGMA','H','BLH','VEG','TREE',\
#            'H2O','USTAR','T_SONIC','ZL','WS','BARE_SOIL','LAI',\
#            'TCC','CAPE','SWC']
#core_var = ['TSS_FILTER','T_SONIC_SIGMA','H','LE','H2O','USTAR',\
#            'ZL','WS','T_SONIC','SWC']

#core_var = ['T_SONIC_SIGMA','ZL2','RHO','USTAR','H']
#core_var = ['H2O_SIGMA','H2O','H','ZL2','BARE_SOIL',\
#            'USTAR','T_SONIC','LE','WS']

core_var.sort()
var_stor = {}
site_var = {'site':[]}

# Run Options
opt_amf          = False  # true includes ameriflux data
opt_neon      = True  # true includes neon data
opt_nighttime = True  # exclude nighttime points
opt_precip      = True  # exclude points with precipitation
opt_freeze      = True  # exclude points below freezing
opt_EBdaily   = False  # True for daily EB, false for point EB (requires G)
opt_EBpct_min = 0
opt_EBpct      = .2    # Energy balance closure residual max
opt_minpoints = 50    # Minimum number of points in a site to be included
opt_minrad    = 0 #100

# Directories
in_dir  = '/home/tsw35/soteria/data/eddy_v2/L2/'
out_dir = 'data/'

# --------- #
# CORE LOOP #
# --------- #
points = {}
sites=os.listdir(in_dir)
for file in sites:
    source='NEON'
    fp_in = h5py.File(in_dir+file,'r')
    n = len(fp_in['TIME'][:])
    points[file]=[n]
    mask = np.ones((n,),dtype=bool)
    print('Evaluating '+file,end='',flush=True)

    mask = mask&(fp_in['H'][:]>-9999)
    mask = mask&(fp_in['LE'][:]>-9999)
    mask = mask&(fp_in['NETRAD'][:]>-9999)    
    mask = mask&(fp_in['NETRAD'][:]>opt_minrad)
    if not opt_EBdaily:
        mask = mask&(fp_in['G'][:]>-9999)
    points[file].append(np.sum(mask))
    
    # check radiative balance
    utc_off = fp_in.attrs['utc_off']
    utc_dt = datetime.timedelta(hours=int(utc_off))
    if (opt_EBdaily):
        time = fp_in['TIME'][:]
        day_old = (datetime.datetime.utcfromtimestamp(time[0]) + utc_dt).day
        H =fp_in['H'][:]
        LE = fp_in['LE'][:]
        netrad = fp_in['NETRAD'][:]
        #mask = mask&(netrad>-9999)
        #mask = mask&(LE>-9999)
        H_sum = 0
        LE_sum = 0
        netrad_sum = 0
        jlist = []
        for j in range(n):
            if not mask[j]:
                continue
            day = (datetime.datetime.utcfromtimestamp(time[j]) + utc_dt).day
            if day_old != day:
                res = netrad_sum-H_sum-LE_sum
                mask[jlist] = mask[jlist] & (np.abs(res/(netrad_sum+.00034))<opt_EBpct)
                H_sum = 0
                LE_sum = 0
                netrad_sum = 0
                jlist = []
                day_old =int(day)
            jlist.append(j)
            H_sum = H_sum+H[j]
            LE_sum = LE_sum+LE[j]
            netrad_sum = netrad_sum + netrad[j]
        points[file].append(np.sum(mask))
    else:
        res = fp_in['NETRAD'][:]-fp_in['G'][:]-fp_in['H'][:]-fp_in['LE'][:]
        mask = mask&(np.abs(res/(fp_in['NETRAD'][:]+.00034))<opt_EBpct)
        mask = mask&(np.abs(res/(fp_in['NETRAD'][:]+.00034))>opt_EBpct_min)
        points[file].append(np.sum(mask))
    
    # CHECK PRECIPITATION
    if (opt_precip):
        mask=mask&(fp_in['P'][:]<=.00001)
        points[file].append(np.sum(mask))

    # CHECK ABOVE FREEZING
    if (opt_freeze):
        try:
            mask=mask&((fp_in['TA'][:]>0)|(fp_in['T_SONIC'][:]>0))
        except:
            mask=mask&(fp_in['T_SONIC'][:]>0)
        points[file].append(np.sum(mask))

    
    # CHECK PRESENCE OF PREDICTORS
    for p in core_var:
        if p == 'H2O_SIGMA':
            mask=mask&(fp_in['qH2O'][:]!=1)
        if (p == 'T_SONIC_SIGMA')&('H2O_SIGMA' not in core_var):
            mask=mask&(fp_in['qT_SONIC'][:]!=1)
        if (p=='TSS_FILTER') or (p=='HSS_FILTER'):
            mask=mask&(fp_in[p][:]>=0)
        mask=mask&(fp_in[p][:]>-9999)
        points[file].append(np.sum(mask))

    # CHECK FOR DAYTIME POINTS
    if opt_nighttime:
        mask=mask&(fp_in['DAY'][:]==1)
        points[file].append(np.sum(mask))
    
    # ELIMINATE IF LESS THAN MINPOINTS
    if np.sum(points[file][-1]) < opt_minpoints:
        mask[:]=False
    points[file].append(np.sum(mask))
    
    # ADD DATA
    for var in fp_in.keys():
        if var == 'canopy_heights':
            continue
        if var == 'HSS_FILTER':
            continue
        if var == 'vertical_wind':
            continue
        if var == 'SWCs':
            continue
        if var not in var_stor.keys():
            var_stor[var]=[]
        var_stor[var].extend(fp_in[var][mask])
    try:
        for var in fp_in['canopy_heights'].attrs.keys():
            if var not in site_var.keys():
                site_var[var]=[]
            site_var[var].extend(np.ones((n,))[mask]*fp_in['canopy_heights'].attrs[var])
    except:
        for var in site_var.keys():
            if 'radius' in var:
                site_var[var].extend(np.ones((n,))[mask]*-9999)
    for var in fp_in.attrs.keys():
        try:
            float(fp_in.attrs[var])
        except Exception:
            continue
        if var == 'igbp':
            continue
        if var not in site_var.keys():
            site_var[var]=[]
        site_var[var].extend(np.ones((n,))[mask]*fp_in.attrs[var])
    
    # ADD DRAG
    if 'CD' not in var_stor.keys():
        var_stor['CD']=[]
    var_stor['CD'].extend((fp_in['USTAR'][mask])**2/(fp_in['WS'][mask])**2)
    
    # FINAL STUFF
    mask_len = np.sum(mask)
    site_var['site'].extend(([file[0:4]]*mask_len))
    print('...COMPLETE')
# generate filename
optstr='QAQCv4'
if opt_amf and opt_neon:
    optstr=optstr+'_both'
elif opt_amf:
    optstr=optstr+'_amf'
elif opt_neon:
    optstr=optstr+'_neon'
if opt_EBdaily:
    optstr=optstr+'_dayEB'+str(opt_EBpct)
else:
    optstr=optstr+'_pntEB'+str(opt_EBpct)
if opt_EBpct_min>0:
    optstr=optstr+'_'+str(opt_EBpct_min)
if opt_nighttime:
    optstr=optstr+'_day'
if opt_precip:
    optstr=optstr+'_rain'
if opt_freeze:
    optstr=optstr+'_frez'
core_var.sort()
for var in core_var:
    if var == 'T_SONIC_SIGMA':
        continue
    optstr=optstr+'_'+var[0:2]

# Output file
fp_out=h5py.File(out_dir+optstr+'.h5','w')
for var in var_stor.keys():
    fp_out.create_dataset(var,data=np.array(var_stor[var][:]))
for var in site_var.keys():
    if var =='site':
        asciiList = [n.encode("ascii", "ignore") for n in site_var[var]]
        fp_out.create_dataset(var,data=(asciiList))
    else:
        fp_out.create_dataset(var,data=np.array(site_var[var][:]))

# Site Report
total_pts=len(var_stor['TIME'][:])
out = '---SITE---|-INIT-|-ELE-|-RAD-|PRCIP|FREZE|'
for var in core_var:
    if len(var)>6:
        out+=var[0:5]+'|'
    else:
        out+=(var+'------')[0:5]+'|'
out+='-DAY-|MNPTS|FINAL|'
print(out)
numsites=0
missing=[]
for site in points.keys():
    init=points[site][0]
    pct = np.array(points[site])/init*100
    final = points[site][-1]
    if final>0:
        numsites+=1
    else:
        missing.append(site)
    print(site+'|',end='')
    for val in pct:
        print("{:5.2f}".format(val)+'|',end='',flush=True)
    print("{:5.2f}".format(final))
print('------------------------------------------------------------------------')
try:
    step_loss =[0 for i in range(len(points['US-Snf.h5'][:]))]
except:
    step_loss =[0 for i in range(len(points['ABBY_L2.h5'][:]))]
for site in points.keys():
    for i in range(len(points[site][:])):
        if points[site][i]==0:
            step_loss[i]+=1
            break
print('LOST-----|',end='')
for i in step_loss:
    print("{:5.0f}".format(i)+'|',end='')

print('')
print('Total Sites Remaining: '+str(numsites))
print('TOTAL points remaining: '+str(len(site_var['site'])))
print('SITES LOST')
for site in missing:
    print(site)
print(optstr+'.h5')


