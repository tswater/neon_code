# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_DP04.py ------------------- #
# ---------------------------------------------------------------- #
# Add the important contents of the main product files to the base #
# h5 including stuff

import os
import netCDF4 as nc
import numpy as np
import h5py
import datetime
import ephem
import rasterio
import csv
import subprocess
from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------------- #
# USER INPUTS AND CONSTANTS #
# ------------------------- #
neon_dir = '/home/tsw35/soteria/data/NEON/dp04/'
base_dir = '../../L2/'

outvar = {'T_SONIC_SIGMA':[],
'H':[],'LE':[],'PA':[],'VPD':[],'TA':[],'USTAR':[],'WS':[],'ZL':[],
'RH':[],'T_SONIC':[],'VPT':[],'RHO':[],'H2O':[],
'H2O_SIGMA':[],'qT_SONIC':[],'qH2O':[]}

units = {'U_SIGMA':'m s-1','V_SIGMA':'m s-1','W_SIGMA':'m s-1',
'T_SONIC_SIGMA':'deg C','H':'W m-2','LE':'W m-2','PA':'kPa',
'VPD':'hPa','TA':'deg C','USTAR':'m s-1','WS':'m s-1','ZL':'N/A',
'H2O_SIGMA':'mmolH2O mol-1','H2O':'mmolH2O mol-1',
'RH':'%','T_SONIC':'deg C'}

desc = {}
# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['JERC']
for site in sites[rank::size]:
	if len(site)>4:
		continue
	print(site+': ',end='',flush=True)
	# dictionary for storing data before loading to h5
	ovar=outvar.copy()
	for k in ovar.keys():
		ovar[k]=[]
	
	# Identify the base filename
	site_files=os.listdir(neon_dir+site)
	dp04_base=site_files[0][0:33]
	
	# Load in the base file
	fp_out=h5py.File(base_dir+site+'_L2.h5','r+')
	time=fp_out['TIME'][:] 
	
	# Initialize site-level constants
	site_cnst={'lat':0,'lon':0,'elev':0,'zd':0,'towH':0,'canH':0}
	lat =0
	lon =0
	elev=0
	zd  =0
	towH=0
	canH=0
	utc_off=0
	
	# ----------------- #
	# LOOP THROUGH TIME #
	# ----------------- #
	old_month=0
	for t in time:
		tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
		if tdt.month!=old_month:
			# Check to see if file exists for this month
			old_month=tdt.month
			try:
				fp_in=h5py.File(neon_dir+site+'/'+dp04_base+\
				  str(tdt.year)+'-'+f'{tdt.month:02}'+'.basic.h5','r')
				_load=True
				print('.',end='',flush=True)
			except Exception:
				_load=False
				dp04_times=[]
				print("'",end='',flush=True)
		else:
			_load=False
		if _load:
			# ----------------------- #
			# LOAD IN DP04 MONTH FILE #
			# ----------------------- #
			for i in range(10):
				try:
					th=str(i)
					fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
						  '0_30m/tempSoni/']['timeBgn'][:]
					break;
				except:
					pass
			tst=fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
					  '0_30m/tempSoni/']['timeBgn'][:]
			
			# Create an array of timestamps
			dp04_times=[]
			for ts in tst:
				dt = datetime.datetime(int(ts[0:4]),int(ts[5:7]),\
					 int(ts[8:10]),int(ts[11:13]),int(ts[14:16]))
				dp04_times.append(dt.replace\
					 (tzinfo=datetime.timezone.utc).timestamp())
			
			# load in other arrays of interest
			TS=fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
						  '0_30m/tempSoni/']['mean'][:]
			qTS=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
						  '0_30m/tempSoni/']['qfFinl'][:]
			TSS=np.sqrt(fp_in['/'+site+'/dp01/data/soni/000_0'+\
							 th+'0_30m/tempSoni/']['vari'][:])
			TA= fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
					  '0_30m/tempAir/']['mean'][:]
			WS= fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_30m/'+\
					  'veloXaxsYaxsErth/']['mean'][:]
			PA= fp_in['/'+site+'/dp01/data/h2oTurb/000_0'+th+\
					  '0_30m/presAtm']['mean'][:]
			TD = fp_in['/'+site+'/dp01/data/h2oTurb/000_0'+th+\
					   '0_30m/tempDew']['mean'][:]
			RH = 100*(np.exp((17.625*TD)/(243.04+TD))/\
					  np.exp((17.625*TA)/(243.04+TA)))
			RH[RH>100]=100
			RH[RH<0]=-9999
			H2O= fp_in['/'+site+'/dp01/data/h2oTurb/000_0'+th+\
					   '0_30m/rtioMoleDryH2o']['mean'][:]
			qH2O= fp_in['/'+site+'/dp01/qfqm/h2oTurb/000_0'+th+\
						'0_30m/rtioMoleDryH2o']['qfFinl'][:]
			H2OS =np.sqrt(np.array(fp_in['/'+site+\
						'/dp01/data/h2oTurb/000_0'+th+\
						'0_30m/rtioMoleDryH2o']['vari'][:]))
			USTAR=fp_in['/'+site+'/dp04/data/fluxMome/turb']\
						['veloFric'][:]
			LE=fp_in['/'+site+'/dp04/data/fluxH2o/turb']['flux'][:]
			H=fp_in['/'+site+'/dp04/data/fluxTemp/turb']['flux'][:]
			
			### SECOND LEVEL COMPUTATIONS ###
			svp = .61121*np.exp((18.678-(TS)/234.5)*((TS)/(257.14+(TS))))
			vpd = svp*(1-RH/100)*10
			vpd[RH<0]=-9999
			e  = svp*RH/100
			r  = .622*e/(PA-e)
			# potential temperature and virtual potential temperature
			pt = (TS+273)*(100/PA)**(2/7)
			vpt= pt*(1+.61*r)
			vpt[PA==-9999]=-9999
			vpt[RH==-9999]=-9999
			vpt[TS==-9999]=-9999
			# air density calculations
			Ra = 286.9
			Rw = 461.5
			rho = ((PA-e)/(Ra*(TS+273))+(e)/(Rw*(TS+273)))*1000
			rho[PA==-9999]=-9999
			rho[RH==-9999]=-9999
			rho[TS==-9999]=-9999
			
			# Load in monthly values (planar fit axis coeff)
			pfangx=fp_in[site].attrs['Pf$AngEnuXaxs'][0]
			pfangy=fp_in[site].attrs['Pf$AngEnuYaxs'][0]
			pfofst=fp_in[site].attrs['Pf$Ofst'][0]
			try:
				pfangx=float(pfangx)
			except Exception:
				pfangx=-9999
			try:
				pfangy=float(pfangy)
			except Exception:
				pfangy=-9999
			try:
				pfofst=float(pfofst)
			except Exception:
				pfofst=-9999
	
			# Load in Constants
			if lat==0:
				try: lat=float(fp_in[site].attrs['LatTow'][0])
				except Exception:pass
			if lon==0:
				try: lon=float(fp_in[site].attrs['LonTow'][0])
				except Exception:pass
			if elev==0:
				try: elev=float(fp_in[site].attrs['ElevRefeTow'][0])
				except Exception:pass
			if zd==0:
				try: zd=float(fp_in[site].attrs['DistZaxsDisp'][0])
				except Exception:pass
			if towH==0:
				try: towH=float(fp_in[site].attrs['DistZaxsTow'][0])
				except Exception:pass
			if canH==0:
				try: canH=float(fp_in[site].attrs['DistZaxsCnpy'][0])
				except Exception:pass
			if utc_off ==0:
				timezones={'MST':-7,'PST/MST':-7.5,'PST':-8,'EST':-5,
						   'CST':-6,'AKST':-9,'AST':-4}
				tz_site = str(fp_in[site].attrs['ZoneTime'][0])[2:-1]
				utc_off = timezones[tz_site]
			
	
		# --------------------- #
		# PERFORM EACH TIMESTEP #
		# --------------------- #
		#get dp04 index
		a=np.where(dp04_times[:]==t)[0]
		if len(a)==0:
			# if this time is not present in dp04 or file does not
			# exist, we fill with empty
			for var in ovar.keys():
				ovar[var].append(-9999)
			continue
		else:
			index=a[0]
		
		# now that we have an index, load all the data
		ovar['T_SONIC_SIGMA'].append(TSS[index])
		ovar['T_SONIC'].append(TS[index])
		ovar['WS'].append(WS[index])
		ovar['RH'].append(RH[index])
		ovar['TA'].append(TA[index])
		ovar['PA'].append(PA[index])
		ovar['H2O'].append(H2O[index])
		ovar['H2O_SIGMA'].append(H2OS[index])
		ovar['qH2O'].append(qH2O[index])
		ovar['qT_SONIC'].append(qTS[index])
		ovar['H'].append(H[index])
		ovar['LE'].append(LE[index])
		ovar['USTAR'].append(USTAR[index])
		ovar['VPD'].append(vpd[index])
		ovar['VPT'].append(vpt[index])
		ovar['RHO'].append(rho[index])
		
		
	# ------------------ #
	# FINAL COMPUTATIONS #
	# ------------------ #
	#ZL
	for k in ovar.keys():
		ovar[k]=np.array(ovar[k])
	n=len(ovar['RHO'])
	if ((towH ==0) or (towH==-9999)):
		ovar['ZL']=np.ones((n,))*-9999
	elif (canH<0):
		ovar['ZL']=np.ones((n,))*-9999
	elif (len(ovar['RH'])<=0)&(len(ovar['VPD'])<=0):
		ovar['ZL']=np.ones((n,))*-9999
	elif len(ovar['USTAR'])<0:
		ovar['ZL']=np.ones((n,))*-9999
	else:
		# moL
		vpt_flux = ovar['H'][:]/ovar['RHO'][:]/1005
		moL = -ovar['USTAR'][:]**3*ovar['VPT'][:]/(.4*9.81*vpt_flux)

		# mask to replace with -9999
		m = ovar['T_SONIC'][:]>-100
		m =m&(ovar['H'][:]>-9999)
		m =m&(ovar['RH'][:]>=0)
		m=m&(ovar['PA'][:]>=0)
		m=m&(ovar['USTAR'][:]>-9999)
		m=m&(~np.isnan(moL))

		#finally ZL
		zL = (towH-zd)/moL
		zL[~m] = -9999
		ovar['ZL']=zL

		

	# -------------------- #
	# CREATE HDF5 DATASETS #
	# -------------------- #
	for key in ovar.keys():
		try:
			fp_out.create_dataset(key,data=np.array(ovar[key][:]))
		except:
			fp_out[key][:]=np.array(ovar[key][:])
		fp_out[key].attrs['missing_value']=-9999
		fp_out[key].attrs['source']='NEON_dp04'
		if key in units.keys():
			fp_out[key].attrs['units']=units[key]
		if key in desc.keys():
			 fp_out[key].attrs['description']=desc[key]
	fp_out.attrs['utc_off']=utc_off
	fp_out.attrs['elevation']=elev
	fp_out.attrs['tow_height']=towH
	fp_out.attrs['canopy_height']=canH
	fp_out.attrs['zd']=zd
	fp_out.attrs['lat']=lat
	fp_out.attrs['lon']=lon
	fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
	print('*',flush=True)
