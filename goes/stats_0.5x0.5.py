from scipy import interpolate
import netCDF4 as nc
import xarray
import numpy as np
import glob
import sys
import rasterio
import os
import pickle
import datetime

#Compute correlation length as lim of variance
def compute_covariance_parameters(Q,pos,lim):
    var = Q[0]
    nQ = Q/var
    f=interpolate.interp1d(nQ, pos, fill_value='extrapolate')
    x10 = f(lim)
    return (var**0.5,x10)

#SAMPLE SCRIPT TO PARALLEL
#Open grid dictionary
file=open('grid0.5.p','rb')
grid=pickle.load(file)
lon_min=grid['lon_min']
lon_max=grid['lon_max']
lat_min=grid['lat_min']
lat_max=grid['lat_max']

min_lat=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
cen_lat=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
max_lat=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
min_lon=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
cen_lon=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
max_lon=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])

#Build arrays of maximum, minumim and central lats and lons
for i in range(0,int(grid['pos_x'].shape[0]-1)):
    cen_lat[:,i]=grid['lat_central']
    min_lat[:,i]=lat_min
    max_lat[:,i]=lat_max
for j in range(0,int(grid['pos_y'].shape[0]-1)):
    cen_lon[j,:]=grid['lon_central']
    min_lon[j,:]=lon_min
    max_lon[j,:]=lon_max

dict={} #Initialize dictionary
dict['covariance_functions']={}
dict['maps']={}

#Define tif as a path to a file, code argument
tif=sys.argv[1]
lst=rasterio.open(tif,'r').read(1) #Open tif file as array
lst_ma=np.ma.fix_invalid(lst,lst==0,fill_value=-9999) #mask out invalid values
lst=np.ma.filled(lst_ma,fill_value=-9999)

#Build pickle file path 
timestr=tif.split('_s')[1][0:7]
date=datetime.datetime(int(timestr[0:4]),1,1,0,0)+datetime.timedelta(days=int(timestr[4:7]))
name='/home/tsw35/tyche/data/GOES/pickles/0.5x0.5_arcdegree/cl_'+str(date)+'.p'
#Limit for valid cell: 25% of -9999
lim=0.25
#Limit for difference of spread to the center: 0.15arcdegree (for 1 and 0.5), 0.05arcdegree (for 0.25)
lim_c=0.15

cl_5=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
cl_10=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
cl_15=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
cl_20=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
STD=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
MEAN=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
P50=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
P25=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
P75=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])
flag=np.zeros([int(grid['pos_y'].shape[0]-1),int(grid['pos_x'].shape[0]-1)])

cell=0
pos_y=-1
for j in grid['pos_y'][:-1]:
    pos_y=pos_y+1
    pos_x=-1
    for i in grid['pos_x'][:-1]:
        pos_x=pos_x+1
        cell=cell+1
        if not (pos_x==pos_y):
            cl_5[int(pos_y),int(pos_x)]=np.nan
            cl_10[int(pos_y),int(pos_x)]=np.nan
            cl_15[int(pos_y),int(pos_x)]=np.nan
            cl_20[int(pos_y),int(pos_x)]=np.nan
            STD[int(pos_y),int(pos_x)]=np.nan
            MEAN[int(pos_y),int(pos_x)]=np.nan
            P25[int(pos_y),int(pos_x)]=np.nan
            P50[int(pos_y),int(pos_x)]=np.nan
            P75[int(pos_y),int(pos_x)]=np.nan
            continue
        
        data=lst[int(j):int(j+20),int(i):int(i+20)]
                
        if (np.sum(data==-9999)/(10*10)>lim) or (0 in data):
            cl_5[int(pos_y),int(pos_x)]=np.nan
            cl_10[int(pos_y),int(pos_x)]=np.nan
            cl_15[int(pos_y),int(pos_x)]=np.nan
            cl_20[int(pos_y),int(pos_x)]=np.nan
            STD[int(pos_y),int(pos_x)]=np.nan
            MEAN[int(pos_y),int(pos_x)]=np.nan
            P25[int(pos_y),int(pos_x)]=np.nan
            P50[int(pos_y),int(pos_x)]=np.nan
            P75[int(pos_y),int(pos_x)]=np.nan
            flag[int(pos_y),int(pos_x)]=1-(np.sum(data<=0)/(20*20)) #1 all cells have data; 0 no cell has data
        else:
            xs=np.linspace(min_lon[int(pos_y),int(pos_x)],max_lon[int(pos_y),int(pos_x)],21) #Limits longitudes
            XC=np.zeros([20,20]) #Initialize cental longitudes for 1arcdegree boxe
            ys=np.linspace(min_lat[int(pos_y),int(pos_x)],max_lat[int(pos_y),int(pos_x)],21) #Limits latitudes
            YC=np.zeros([20,20]) #Initialize cental latitudes for 1arcdegree boxes
            #Fill central lats and lons
            for i in range(0,20):
                x_cen=np.mean([xs[int(i)],xs[int(i)+1]])
                y_cen=np.mean([ys[int(i)],ys[int(i)+1]])
                XC[i,:]=x_cen
                YC[:,i]=y_cen
            #Flatten arrays
            r_data=np.ravel(data)
            r_X=np.ravel(XC)
            r_Y=np.ravel(YC)
            #Mask out -9999
            m1=r_data!=-9999
            #filtered versions of position arrays
            fil_X=r_X[m1]
            fil_Y=r_Y[m1]
            #filtered version of data array
            fil_data=r_data[m1]
            #central latitudes and longitudes
            mean_X=np.mean(fil_X)
            mean_Y=np.mean(fil_Y)
            if (abs(cen_lon[int(pos_y),int(pos_x)]-mean_X)>lim_c) and (abs(cen_lat[int(pos_y),int(pos_x)]-mean_Y)>lim_c):
                cl_5[int(pos_y),int(pos_x)]=np.nan #Not enough data
                cl_10[int(pos_y),int(pos_x)]=np.nan #Not enough data
                cl_15[int(pos_y),int(pos_x)]=np.nan #Not enough data
                cl_20[int(pos_y),int(pos_x)]=np.nan #Not enough data
                STD[int(pos_y),int(pos_x)]=np.std(r_data[m1])
                MEAN[int(pos_y),int(pos_x)]=np.mean(r_data[m1])
                P25[int(pos_y),int(pos_x)]=np.percentile(r_data[m1],25)
                P50[int(pos_y),int(pos_x)]=np.percentile(r_data[m1],50)
                P75[int(pos_y),int(pos_x)]=np.percentile(r_data[m1],75)
                flag[int(pos_y),int(pos_x)]=1-(np.sum(data==-9999)/(20*20))
            else:
                h = (((fil_X[:,np.newaxis] - (fil_X).T)**2 + (fil_Y[:,np.newaxis] - (fil_Y).T)**2)**0.5)*111000 #m (111km/arcdegree)
                #Reshape to 1d array
                h = h.reshape(h.size)
                #Define bin edges
                bin_edges=np.logspace(np.log10(2674),np.log10(38000),num=25,base=10) #LOG SPACE
                #Compute the center of each bin
                hd = (bin_edges[1:] + bin_edges[0:-1])/2
                #Compute mean of values
                mu = np.mean(fil_data)
                a=(fil_data[:,np.newaxis].T-mu)*(fil_data[:,np.newaxis]-mu)
                a=a.reshape(a.size)
                #Compute sample covariance function
                Q=[]
                Q.append(np.std(fil_data)**2) #Add var for scale=0m
                hdn=[]
                hdn.append(0)
                for i in range(hd.size):
                    if i==0:m=(h >= bin_edges[i]) & (h < bin_edges[i+1])
                    else:m = (h > bin_edges[i]) & (h <= bin_edges[i+1])
                    if np.sum(m) < 2:
                        hdn.append(hd[i])
                        Q.append(np.nan)
                        continue
                    tmp=np.mean(a[m])
                    if (tmp<0):tmp=0 #Enforce positive covariance
                    hdn.append(hd[i])
                    Q.append(tmp)
                m_nan=~np.isnan(Q)
                Q_nan=np.array(Q)[m_nan]
                hdn_nan=np.array(hdn)[m_nan]
                (std,x5) = compute_covariance_parameters(list(Q_nan),list(hdn_nan),0.05)
                if x5<0:x5=np.nan
                cl_5[int(pos_y),int(pos_x)]=x5
                (std,x10) = compute_covariance_parameters(list(Q_nan),list(hdn_nan),0.10)
                if x10<0:x10=np.nan
                cl_10[int(pos_y),int(pos_x)]=x10
                (std,x15) = compute_covariance_parameters(list(Q_nan),list(hdn_nan),0.15)
                if x15<0:x15=np.nan
                cl_15[int(pos_y),int(pos_x)]=x15
                (std,x20) = compute_covariance_parameters(list(Q_nan),list(hdn_nan),0.20)
                if x20<0:x20=np.nan
                cl_20[int(pos_y),int(pos_x)]=x20
                STD[int(pos_y),int(pos_x)]=np.std(r_data[m1])
                MEAN[int(pos_y),int(pos_x)]=np.mean(r_data[m1])
                P25[int(pos_y),int(pos_x)]=np.percentile(r_data[m1],25)
                P50[int(pos_y),int(pos_x)]=np.percentile(r_data[m1],50)
                P75[int(pos_y),int(pos_x)]=np.percentile(r_data[m1],75)
                flag[int(pos_y),int(pos_x)]=1-(np.sum(data==-9999)/(20*20))
                dict['covariance_functions'][str(cell)]={}
                dict['covariance_functions'][str(cell)]['pos']=hdn
                dict['covariance_functions'][str(cell)]['Q']=Q
                #print(cell,pos_x,pos_y,x5,x10,x15,x20,STD[int(pos_y),int(pos_x)],MEAN[int(pos_y),int(pos_x)],P25[int(pos_y),int(pos_x)],flush=True)

dict['maps']['cl_5']=cl_5
dict['maps']['cl_10']=cl_10
dict['maps']['cl_15']=cl_15
dict['maps']['cl_20']=cl_20
dict['maps']['std']=STD
dict['maps']['mean']=MEAN
dict['maps']['p25']=P25
dict['maps']['p50']=P50
dict['maps']['p75']=P75
dict['maps']['Q_flag']=flag
dict['maps']['lon_min']=min_lon
dict['maps']['lon_max']=max_lon
dict['maps']['lat_min']=min_lat
dict['maps']['lat_max']=max_lat
dict['maps']['lon_central']=cen_lon
dict['maps']['lat_central']=cen_lat

#Save pickle file
pickle.dump(dict,open('%s'%name,'ab'))
