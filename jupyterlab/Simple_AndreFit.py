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
fp=0

# %%
fp.close()

# %%
fp = h5py.File('data/QAQCv4_neon_pntEB0.2_day_rain_frez_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5','r')

# %%
pred = ['H','USTAR','WS','VEG','TREE','H2O','T_SONIC','BARE_SOIL','LAI','ZL','BLH','TCC','CAPE','SWC']
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

n = len(fp['H'][:])
n2 = len(pred)
result_old=(fp['T_SONIC_SIGMA'][:])
#result=(fp['T_SONIC_SIGMA'][:])
result=(fp['TSS_FILTER'][:])
mins={'WS':0,
      'ZL':-3,
      'USTAR':0,
      'CAPE':0,
      'H':-200}
maxs={'WS':20,
      'ZL':1.5,
      'USTAR':2,
      'CAPE':1000,
      'H':650}
rmin = 0
rmax = 1.275
data = np.zeros((n2,n))
i = 0
j = 0
fig = plt.figure(figsize=(12,9))
for x in pred:
    print(x)
    data[i,:]=fp[x][:]
    i = i+1
data = np.transpose(data)

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
# Generate weights
numpoints = {}
for i in range(len(fpsites)):
    site = fpsites[i]
    if site not in numpoints.keys():
        numpoints[site] = 1
    else:
        numpoints[site] = numpoints[site]+1
n = len(sites)
for e in numpoints.keys():
    numpoints[e]=1/numpoints[e]
weights = []
for i in range(len(fpsites)):
    site = fpsites[i]
    weights.append(numpoints[site])
indexes = np.linspace(0,len(fpsites)-1,len(fpsites))
weights=np.array(weights)

# %%
### FIT BY ITERATION ####

import warnings; warnings.simplefilter('ignore')
zL = fp['ZL2'][:]
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
m = (np.abs(fp['H'][:])>0)&(zL<0)
c_vals = np.linspace(.2,2,201)
H = fp['H'][:]/1005/rho
o_Tsig = result.copy()[m]
a=6
best={'nRMSE':1,'nBias':1,'c':0}
for c in c_vals:
    andre_RHS = c**2*(-zL)**(-.66)
    andre_RHS[zL>=0]=a
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
    bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)
    rmse2 = metrics.mean_squared_error(o_Tsig, andre_Tsig,squared=False)/np.mean(o_Tsig)
    if rmse2 < best['nRMSE']:
        best['nRMSE']=rmse2
        best['nBias']=bias_2
        best['c']=c

# %%
# plot fit by iteration
andre_RHS = best['c']**2*(-zL)**(-.66)
#andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
plt.figure()
plt.hexbin(o_Tsig,andre_Tsig,gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.25,0,1.25])
#plt.scatter(o_Tsig,andre_Tsig)
plt.plot([0,1.25],[0,1.25],"k--",linewidth=3)
plt.title("Equation 1 Validation ")
plt.xlabel("Eddy Flux Tower Observed Value $\sigma_T$ (K)")
plt.ylabel("EQ 1 Predicted Value $\sigma_T$ (K)")
bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)/np.mean(o_Tsig)
print('Bias: '+str(best['nBias']/np.mean(o_Tsig)))
print('nRMSE: '+str(best['nRMSE']))
print('c: '+str(best['c']))

# %%
### Site by Site Fit by Iteration ###
c_s={}
nrmse_s={}
nbias_s={}
ormse_s={}
obias_s={}
import warnings; warnings.simplefilter('ignore')
zL = fp['ZL2'][:]
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
m_0 = (np.abs(fp['H'][:])>0)&(zL<0)
c_vals = np.linspace(.2,2,201)
H = fp['H'][:]/1005/rho
for site in sites:
    print(site)
    m=(fpsites==site)&(m_0)
    o_Tsig = result.copy()[m]
    best={'nRMSE':1,'nBias':1,'a':0,'b':0}
    for c in c_vals:
        andre_RHS=c**2*(-zL)**(-.66)
        andre_RHS[zL>=0]=a
        andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
        bias_2 = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)
        rmse2 = np.sqrt(metrics.mean_squared_error(o_Tsig, andre_Tsig)/np.mean(o_Tsig))
        if rmse2 < best['nRMSE']:
            best['nRMSE']=rmse2
            best['nBias']=bias_2
            best['c']=c
    c_s[site]=best['c']
    nrmse_s[site]=best['nRMSE']
    nbias_s[site]=best['nBias']
    a=4
    c=.95
    andre_RHS = c**2*(-zL)**(-.66)
    andre_RHS[zL>=0]=a
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))[m]
    obias_s[site] = np.sum(andre_Tsig-o_Tsig)/len(o_Tsig)/np.mean(o_Tsig)
    ormse_s[site] = np.sqrt(metrics.mean_squared_error(o_Tsig, andre_Tsig)/np.mean(o_Tsig))

# %%

# %%
### NLCD STUFF ADDD ###
# ------------ #
# SITE BY SITE #
# ------------ #
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Frst',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wood Wet',95:'Herb Wet'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue'}

nlcd_dom = fp['nlcd_dom'][:]
sites =[]
fpsites = fp['site'][:]
npoints =0
apoints =0
nsites=0
asites=0
nlcds = {}
site_nlcds={}
nlcd_sums={}
site_nlcd_dom={}
i=0
for point in fpsites:
    if point not in sites:
        sites.append(point)
        site_nlcds[point]=[]
        for k in fp.keys():
            if 'nlcd' in str(k):
                if k=='nlcd_dom':
                    continue
                site_nlcds[point].append(fp[k][i])
                if str(k) not in nlcds.keys():
                    nlcds[str(k)]=[]
                    nlcd_sums[str(k)]=0
                nlcds[str(k)].append(fp[k][i])
                nlcd_sums[str(k)]=nlcd_sums[str(k)]+fp[k][i]
    if 'x-' in str(point):
        npoints=npoints+1
    else:
        apoints=apoints+1
    i = i+1
for site in sites:
    site_nlcd_dom[site]=nlcd_dom[fpsites==site][0]
    if 'x-' in str(site):
        nsites=nsites+1
    else:
        asites=asites+1
for k in nlcd_sums.keys():
    print(k+": "+str(nlcd_sums[k]))

# %%

# %%
delta_rmse=np.array(list(nrmse_s.values()))-np.array(list(ormse_s.values()))
delta_bias=np.abs(list(nbias_s.values()))-np.abs(list(obias_s.values()))
names=[]
clist=list(c_s.values())
plot_colors=[]
for k in nrmse_s.keys():
    names.append(k)
    plot_colors.append(class_colors[site_nlcd_dom[k]])
plt.figure(figsize=(15,4))
bar1=plt.bar(names,delta_rmse*100,color=plot_colors)
plt.xticks(rotation=45)
i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height-.7,str(round(clist[i]*100)/100), ha='center', va='bottom',color='blue')
    i=i+1
plt.ylim(-16,0)
plt.ylabel('Change in Error after Fitting')
plt.title('Change in Error after fitting A and B in André for each site')

# %%
plt.figure()
#plt.scatter(list(a_s.values()),list(b_s.values()),c=list(nrmse_s.values()),cmap='bwr')
plt.scatter(np.log(list(canh.values())),np.array(list(c_s.values())),c=plot_colors,s=75)
plt.ylabel('Change in C1 when fitted')
plt.xlabel('Canopy Height')
plt.title('Deviation of best fit C1 from .95 vs Error')

# %%
canh_x={}
canh_keys=[]
canh={}
for k in fp.keys():
    if 'radius' in k:
        canh_keys.append(k)
for k in canh_keys:
    canh_x[k]={}
for site in sites:
    m_site = fpsites[:]==site
    canh[site]=fp['canopy_height'][m_site][0]
    for k in canh_keys:
        canh_x[k][site]=fp[k][:][m_site][0]

# %%
plt.figure(figsize=(15,15))
for i in range(16):
    k=canh_keys[i]
    plt.subplot(4,4,i+1)
    print(np.mean(list(canh_x[k].values())))
    plt.scatter(np.array(list(canh_x[k].values())),(np.abs(.95-np.array(list(c_s.values())))),c=plot_colors,s=75)
    plt.title(k)
    plt.ylim(0,.3)

# %%

# %%
plt.figure(figsize=(15,4))
bar1=plt.bar(names,delta_bias*100,color=plot_colors)
plt.xticks(rotation=45)
i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height-1.5,str(int(round(alist[i]))), ha='center', va='bottom',color='blue')
    plt.text(rect.get_x() + rect.get_width()/2.0, height-2.7,str(int(round(blist[i]))), ha='center', va='bottom',color='green')
    i=i+1
plt.ylim(-20,3)
plt.ylabel('Change in Absolute Bias after Fitting')
plt.title('Change in Absolute Bias after fitting A and B in André for each site')

# %%

# %%

# %%
data=np.zeros((2,len(sites)))
results=np.zeros((2,len(sites)))
i=0
for site in sites:
    nlst=site_nlcds[site]
    data[0,i]= nlst[7]+nlst[8]+nlst[9]+nlst[18] #7,8,9,18
    data[1,i]= nlst[10]+nlst[11]+nlst[12]+nlst[0]+nlst[2]+nlst[3]+nlst[16]+nlst[17]+nlst[19] #10,11,12,0,2,3,16,17,19
    results[0,i]=a_s[site] #a
    results[1,i]= b_s[site]#b
    i=i+1

# %%
clf = sklearn.ensemble.RandomForestRegressor(random_state=1,oob_score=True)
b_test=results[1,:]
a_test=results[0,:]
data2=np.transpose(data)
clf.fit(data2,b_test)
b_pred = clf.predict(data2)
clf_bscore=clf.oob_score_
b_rmse=np.sqrt(metrics.mean_squared_error(b_test, b_pred))/np.mean(b_test)
print(b_rmse)
print(clf_bscore)

clf.fit(data2,a_test)
a_pred = clf.predict(data2)
clf_ascore=clf.oob_score_
a_rmse=np.sqrt(metrics.mean_squared_error(a_test, a_pred))/np.mean(a_test)
print(a_rmse)
print(clf_ascore)

plt.figure()
plt.scatter(a_test,a_pred)
plt.title('PREDICT A -- OOBscore:'+f'{clf_ascore:0.2f}'+'   nrmse:'+f'{a_rmse:.2f}')
plt.xlabel('Observed Best Fit')
plt.ylabel('RF Predicted Value')

plt.figure()
plt.scatter(b_test,b_pred)
plt.title('PREDICT B -- OOBscore:'+f'{clf_bscore:0.2f}'+'   nrmse:'+f'{b_rmse:.2f}')
plt.xlabel('Observed Best Fit')
plt.ylabel('RF Predicted Value')

# %%
print(nbias_s[b'ORNL'])
print(obias_s[b'ORNL'])

# %%
### CHECK ERROR WITH PREDICTED A,B 
i=0
n2_rmse={}
n2_bias={}
for site in sites:
    a=a_pred[i]
    b=b_pred[i]
    i=i+1
    m=fpsites==site
    zL=fp['ZL'][m]
    rho = fp['RHO'][m]
    H = fp['H'][m]/1005/rho
    ustar=fp['USTAR'][m]
    andre_RHS = a*(1-b*zL)**(-.66)
    andre_RHS[zL>=0]=a
    andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))
    o_Tsig = fp['TSS_FILTER'][m]
    n2_rmse[site]=np.sqrt(metrics.mean_squared_error(o_Tsig, andre_Tsig)/np.mean(o_Tsig))
    n2_bias[site]=np.sum(andre_Tsig-(o_Tsig))/len(o_Tsig)/np.mean(andre_Tsig)
plt.figure(figsize=(15,4))
plt.bar(sites,n2_rmse.values(),alpha=.5)
plt.bar(sites,ormse_s.values(),alpha=.5)
plt.xticks(rotation=45)


# %%
### SET UP RF to PREDICT a,b,c BASED ON pred

# %%
### FIT METHOD ####

# %%
def andre_tsig(X,a,b,c):
    # H must be kinheat
    H=X[0,:]
    USTAR=X[1,:]
    zL=X[2,:]
    HUSTAR=H**2/USTAR**2
    result=HUSTAR*a*(1-b*ZL)**(-2/3)+c
    result[ZL>0]=HUSTAR*a+c
    return np.sqrt(result)

kinheat = fp['H'][:]/fp['RHO'][:]/1005
import scipy.optimize
popt,pcov = scipy.optimize.curve_fit(aHb,ORNL_H[ORNL_H>0],ORNL_TSS[ORNL_H>0])

# %%

# %%
