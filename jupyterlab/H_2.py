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

# %%
fp.close()

# %%
fp = h5py.File('data/QAQCv4_neon_pntEB0.2_day_rain_frez_BA_BL_CA_H_H2_H2_H2_LA_SW_TC_TR_T__US_VE_WS_ZL.h5','r')

# %%
print(fp.keys())

# %%
pred = ['H','LE','USTAR','WS','BLH','VEG','TREE','H2O','T_SONIC','ZL','BARE_SOIL','LAI','TCC','CAPE','SWC']
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
result_old=(fp['H2O_SIGMA'][:])
result=(fp['H2OS_FILTER'][:])

# %%
plt.hist(fp['H2O_SIGMA'][:],bins=100)
plt.title('')

# %%
plt.hist(fp['H2OS_FILTER'][:],bins=100)
plt.title('')

# %%
mins={'WS':0,
      'ZL':-3,
      'USTAR':0,
      'CAPE':0,
      'LE':-100}
maxs={'WS':20,
      'ZL':1.5,
      'USTAR':2,
      'CAPE':1000,
      'LE':650}
rmin = 0
rmax = 2.5
data = np.zeros((n2,n))

i = 0
j = 0
fig = plt.figure(figsize=(12,9))
for x in pred:
    print(x)
    if(j==9):
        fig.suptitle('Direct Comparison of Predictors vs $\sigma_T$',fontsize=20)
        plt.subplots_adjust(hspace=.25,wspace=.15)
        fig=plt.figure(figsize=(12,9))
        j=0
        
    data[i,:]=fp[x][:]
    # make a quick plot
    plt.subplot(3,3,j+1)
    try:
        plt.hexbin(data[i,:],result,gridsize=200,cmap='terrain',mincnt=1,extent=(mins[x],maxs[x],rmin,rmax))
    except Exception:
        plt.hexbin(data[i,:],result,gridsize=200,cmap='terrain',mincnt=1)
    #plt.hexbin(data[i,:],result,gridsize=100,cmap='terrain',mincnt=1,extent=(mins[x],maxs[x],rmin,rmax))
    #plt.scatter(data[i,:],result[:],alpha=.01)
    #plt.title(x+' vs T_SONIC_SIGMA')
    plt.title(names[x])
    #plt.ylabel('Sigma T (K)')
    i = i+1
    j = j+1
fig.suptitle('Direct Comparison of Predictors vs $\sigma_T$',fontsize=20)
plt.subplots_adjust(hspace=.25,wspace=.15)
data = np.transpose(data)

# %%
#m = np.zeros((len(fpsites),),dtype=bool)
#for i in range(len(fpsites)):
#    m[i]=(i in in_test)
zL = fp['ZL'][:]
#m = m&(zL>0)
#m = m & (fp['H'][:]>0)
#m = m & (isNeon)
m = np.ones((n,),dtype=bool)
a = 4
b = 8.3
delta =0
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
LE = fp['LE'][:]/2472/rho
andre_RHS = a*(1-b*zL)**(-.66)
andre_RHS[zL>=0]=a
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/LE**2))
o_Tsig = result.copy()*18/29
plt.figure()
plt.hexbin(o_Tsig[m]-delta,andre_Tsig[m],gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.5,0,1.5])
#plt.scatter(o_Tsig,andre_Tsig)
limmin = np.min([np.min(o_Tsig),np.min(andre_Tsig)])
limmax = np.max([np.max(o_Tsig),np.max(andre_Tsig)])
plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3)
plt.xlim(0,1.5)
plt.ylim(0,1.5)
plt.title("Equation 1 Validation -- Filtered",fontsize=18)
plt.xlabel("Eddy Flux Tower Observed Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
plt.ylabel("EQ 1 Predicted Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
plt.figure()
bias_2 = np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])
print('Bias: '+str(bias_2/np.mean(o_Tsig[m]-delta)))
print(np.sqrt(metrics.mean_squared_error(o_Tsig[m]-delta, andre_Tsig[m])/np.mean(o_Tsig[m]-delta)))
print(np.mean(o_Tsig[m]-andre_Tsig[m]))
print()
plt.hexbin(result_old[m]*18/29-delta,andre_Tsig[m],gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.5,0,1.5])
plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3)
plt.xlim(0,1.5)
plt.ylim(0,1.5)
plt.title("Equation 1 Validation -- Unfiltered",fontsize=18)
plt.xlabel("Eddy Flux Tower Observed Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
plt.ylabel("EQ 1 Predicted Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
bias_2 = np.sum(andre_Tsig[m]-(result_old[m]*18/29-delta))/len(o_Tsig[m])
print('Bias: '+str(bias_2/np.mean(result_old[m]-delta)))
print(np.sqrt(metrics.mean_squared_error(result_old[m]*18/29-delta, andre_Tsig[m])/np.mean(fp['H2O_SIGMA'][m]-delta)))
print(np.mean(result_old[m]*18/29-andre_Tsig[m]))

# %%
plt.hexbin(result,fp['T_SONIC_SIGMA'][:],gridsize=100,cmap='terrain',mincnt=1,extent=[0,1.5,0,1.5])
plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3)
plt.xlim(0,1.5)
plt.ylim(0,1.5)

# %%
# ------------ #
# SITE BY SITE #
# ------------ #
sites =[]
fpsites = fp['site'][:]
for point in fpsites:
    if point not in sites:
        sites.append(point)
TSS=fp['T_SONIC_SIGMA'][:]

# %%
from sklearn.linear_model import LinearRegression
def tilt(obs,pr):
    model = LinearRegression(fit_intercept=False)
    model.fit(obs.reshape((-1, 1)),pr)
    return model.coef_[0]


# %%
plt.figure(figsize=(15,15))
i=1
errors={}
biases={}
tower_height={}
tilts={}
canh={}
precips={}
temps={}
for site in sites:
    m_site = fpsites[:]==site
    if i == 26:
        i=1
        plt.subplots_adjust(wspace=.3,hspace=.25)
        plt.figure(figsize=(15,15))
    plt.subplot(5,5,i)
    #m = np.ones((n,),dtype=bool)
    m=m_site
    errors[site]=np.sqrt(metrics.mean_squared_error(o_Tsig[m]-delta, andre_Tsig[m])/np.mean(o_Tsig[m]-delta))
    biases[site]=np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])/np.mean(o_Tsig[m]-delta)
    temps[site]=fp['mean_temp'][m][0]
    precips[site]=fp['mean_precip'][m][0]
    tower_height[site]=fp['tow_height'][m][0]
    canh[site]=fp['canopy_height'][m][0]
    plt.hexbin(o_Tsig[m]-delta,andre_Tsig[m],gridsize=40,cmap='terrain',mincnt=1,extent=[0,1,0,1])
    limmin = np.min([np.min(o_Tsig),np.min(andre_Tsig)])
    limmax = np.max([np.max(o_Tsig),np.max(andre_Tsig)])
    plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3,alpha=.7)
    plt.xlim(0,1)
    plt.ylim(0,1)
    tilts[site]=tilt(o_Tsig[m],andre_Tsig[m])
    plt.title(str(site)[2:-1]+" "+f'{tilts[site]:1.2f}')
    i=i+1
plt.subplots_adjust(wspace=.3,hspace=.25)

# %%
# PLOT TOW_HEIGHT vs SLOPE
plt.figure()
plt.scatter(tower_height.values(),tilts.values())
plt.xlabel('Tower Height')
plt.ylabel('Slope Best Fit')
plt.title('Tower Height vs Slope',fontsize=16)
# PLOT TOW_HEIGHT vs ERROR
plt.figure()
plt.scatter(tower_height.values(),errors.values())
plt.xlabel('Tower Height')
plt.ylabel('Errors')
plt.title('Tower Height vs Error',fontsize=16)
# PLOT SLOPE vs ERROR
plt.figure()
plt.scatter(tilts.values(),errors.values())
plt.xlabel('Slope Best Fit')
plt.ylabel('Errors')
plt.title('Slope vs Error',fontsize=16)
# PLOT TOW_HEIGHT vs BIAS
plt.figure()
plt.scatter(tower_height.values(),biases.values())
plt.xlabel('Tower Height')
plt.ylabel('Biases')
plt.title('Tower Height vs Bias',fontsize=16)
# PLOT CAN_H vs ERROR
plt.figure()
plt.scatter(canh.values(),errors.values())
plt.xlabel('Canopy Height')
plt.ylabel('Errors')
plt.title('Canopy Height vs Error',fontsize=16)
# PLOT CAN_H vs ERROR
plt.figure()
plt.scatter(np.array(list(tower_height.values()))-np.array(list(canh.values())),errors.values())
plt.xlabel('Difference Height')
plt.ylabel('Errors')
plt.title('Tower - Canopy Height vs Error',fontsize=16)
# PLOT mean_precip vs ERROR
plt.figure()
plt.scatter(precips.values(),errors.values())
plt.xlabel('Precip')
plt.ylabel('Errors')
plt.title('Mean Precip vs Errror',fontsize=16)
# PLOT mean_temp vs ERROR
plt.figure()
plt.scatter(temps.values(),errors.values())
plt.xlabel('Temp')
plt.ylabel('Errors')
plt.title('Mean Temp vs Error',fontsize=16)
# PLOT mean_precip vs SLOPE
plt.figure()
plt.scatter(precips.values(),tilts.values())
plt.xlabel('Precip')
plt.ylabel('Slope')
plt.title('Mean Precip vs Slope Best Fit',fontsize=16)
# PLOT mean_temp vs SLOPE
plt.figure()
plt.scatter(temps.values(),tilts.values())
plt.xlabel('Temp')
plt.ylabel('Slope')
plt.title('Mean Temp vs Slope Best Fit',fontsize=16)


# %%
plt.figure()
ustar=fp['USTAR'][:]
humid_flux=fp['LE'][:]/(2472)
heat_flux =fp['H'][:]/1005/fp['RHO'][:]

'''
m1=(np.abs(humid_flux)<.1)&(humid_flux>0)
humid_flux[m1]=.1
m2=(np.abs(humid_flux)<.1)&(humid_flux<=0)
humid_flux[m2]=-.1

m1=(np.abs(heat_flux)<.1)&(heat_flux>0)
heat_flux[m1]=.1
m2=(np.abs(heat_flux)<.1)&(heat_flux<=0)
heat_flux[m2]=-.1
'''

#m3=(np.abs(humid_flux)>.1)&(np.abs(heat_flux)>.1)
m3=(np.abs(fp['LE'][:])>25)&(np.abs(fp['H'][:])>25)
m3=m3&(fp['LE'][:]>25)&(np.abs(fp['H'][:])>25)
m3=m3&(~np.isnan(fp['TSS_FILTER'][:]))

water_andre = (fp['H2OS_FILTER'][:]*18/28.97)**2*ustar**2/(humid_flux)**2
heat_andre  = fp['TSS_FILTER'][:]**2*ustar**2/(heat_flux)**2
print(np.mean(humid_flux))
print(np.mean(heat_flux))

print(np.mean(fp['H2O_SIGMA'][m3]*18/28.97/humid_flux[m3])**2)
print(np.mean(fp['T_SONIC_SIGMA'][m3]/heat_flux[m3])**2)

print(np.median(fp['H2O_SIGMA'][m3]*18/28.97/humid_flux[m3])**2)
print(np.median(fp['T_SONIC_SIGMA'][m3]/heat_flux[m3])**2)
import scipy.stats
r=scipy.stats.pearsonr(water_andre[m3],heat_andre[m3])
print(r)
r2 =scipy.stats.spearmanr(water_andre[m3],heat_andre[m3])
print(r2)
print(np.sum(water_andre<30))
plt.hexbin(water_andre[m3],heat_andre[m3],gridsize=200,mincnt=1,cmap='terrain',extent=(0,15,0,15))
plt.plot([0,15],[0,15],'k--')
plt.xlabel("Moisture: $\overline{q'}^2$/$(w'q')^2$",fontsize=14)
plt.ylabel("Heat: $\overline{\\theta'}^2$/$(w'T')^2$",fontsize=14)
plt.title('Comparing André Moisture and Heat Flux Terms',fontsize=16)

# %%
i = 1
highlight=[b'TREE',b'x-DELA',b'x-TALL',b'x-BART',b'x-NOGP',\
           b'TEAK',b'x-JORN',b'x-NIWO',b'x-SRER',b'x-MLBS',\
           b'x-ABBY',b'x-STER',b'x-CLBJ',b'x-UKFS',b'x-KONA']
highlight=[b'TREE',b'TEAK',b'ABBY',\
           b'DELA',b'JORN',b'STER',\
           b'TALL',b'NIWO',b'CLBJ',\
           b'BART',b'SRER',b'UKFS',\
           b'NOGP',b'MLBS',b'KONA']
fig = plt.figure(figsize=(10,15))
for site in highlight:
    m_site = fpsites[:]==site
    if i == 26:
        i=1
        plt.figure(figsize=(15,15))
    plt.subplot(5,3,i)
    plt.hexbin(water_andre[m_site&m3],heat_andre[m_site&m3],gridsize=100,cmap='terrain',mincnt=1,extent=(0,15,0,15))
    plt.plot([0,15],[0,15],'k--')
    plt.title(str(site)[2:-1])
    i = i+1
plt.subplots_adjust(hspace=.25,wspace=.15)
fig.text(0.5,0.92,'Comparing André Moisture and Heat Terms by Site',fontsize=26,ha='center')
fig.text(0.5, 0.08, "Moisture: $\overline{q'}^2$/$(w'q')^2$",fontsize=20, ha='center')
fig.text(0.05, 0.5, "Heat: $\overline{\\theta'}^2$/$(w'T')^2$",fontsize=20, va='center', rotation='vertical')

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

# %%
X_train, X_test, y_train, y_test, w_train,w_test = train_test_split(data,result,weights, test_size=0.2, random_state=0)

# %%
# ------------- #
# RANDOM FOREST #
# ------------- #

clf = sklearn.ensemble.RandomForestRegressor(random_state=1,n_estimators=100,oob_score=True)
clf.fit(X_train,y_train,w_train)
y_pred = clf.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(clf.oob_score_)

# %%
plt.figure()
#plt.scatter(y_test,y_pred,alpha=.01)
plt.hexbin(y_test,y_pred,cmap='terrain',mincnt=1)
limmin = np.min([np.min(y_test),np.min(y_pred)])
limmax = np.max([np.max(y_test),np.max(y_pred)])
plt.plot([limmin,limmax],[limmin,limmax],"k:")
plt.title("RF Validation -- Filtered")
plt.xlabel("Observed Value Sigma_T (K)")
plt.ylabel("Predicted Value Sigma_T (K)")
#plt.xlim(0,5)
#plt.ylim(0,5)

#make a names list
namelist=[]
for x in pred:
    namelist.append(names[x])

# sort the feature importance
fIMP = clf.feature_importances_
points = fIMP
fIMP_names={}
for i in range(len(namelist)):
    fIMP_names[points[i]]=namelist[i]
points[::-1].sort()
names2=[]
for i in range(len(namelist)):
    names2.append(fIMP_names[points[i]])
    
# plot
plt.figure()
plt.bar(names2,points)
plt.title("RF Feature Importance")
plt.xlabel("Predictor")
plt.ylabel("Importance")
plt.xticks(rotation=45)
bias = np.sum(y_pred-y_test)/len(y_pred)
print('Bias: '+str(bias/np.mean(y_test)))
print('Normalized RMSE: '+str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.mean(y_test)))
print('OOB Score: '+ str(clf.oob_score_))

# %%
from sklearn.ensemble import GradientBoostingRegressor

# %%
# ----------------- #
# GRADIENT BOOSTING #
# ----------------- #
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train, w_train)
y_pred_g=reg.predict(X_test)
sc = reg.score(X_test,y_test)

# %%
plt.figure()
#plt.scatter(y_test,y_pred,alpha=.01)
plt.hexbin(y_test,y_pred_g,cmap='terrain',mincnt=1)
limmin = np.min([np.min(y_test),np.min(y_pred)])
limmax = np.max([np.max(y_test),np.max(y_pred)])
plt.plot([limmin,limmax],[limmin,limmax],"k:")
plt.title("Boosting Validation ")
plt.xlabel("Observed Value Sigma_T (K)")
plt.ylabel("Predicted Value Sigma_T (K)")

#make a names list
namelist=[]
for x in pred:
    namelist.append(names[x])

# sort the feature importance
fIMP = reg.feature_importances_
points = fIMP
fIMP_names={}
for i in range(len(namelist)):
    fIMP_names[points[i]]=namelist[i]
points[::-1].sort()
names2=[]
for i in range(len(namelist)):
    names2.append(fIMP_names[points[i]])
print(np.array(names2).shape)
print(points.shape)
    
# plot
plt.figure()
plt.bar(names2,points)
plt.title("Boosting Feature Importance")
plt.xlabel("Predictor")
plt.ylabel("Importance")
plt.xticks(rotation=45)
bias = np.sum(y_pred_g-y_test)/len(y_pred_g)
print('Bias: '+str(bias/np.mean(y_test)))
print('Normalized RMSE: '+str(np.sqrt(metrics.mean_squared_error(y_test, y_pred_g))/np.mean(y_test)))
print('Score: '+ str(sc))

# %%
# ------------ #
# SITE BY SITE #
# ------------ #
sites =[]
fpsites = fp['site'][:]
nlcd_dom = fp['nlcd_dom'][:]
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
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Frst',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wood Wet',95:'Herb Wet'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue'}
plt.figure(figsize=(12,4))
plt.bar(list(class_names.values()),list(nlcd_sums.values()),color=class_colors.values())
plt.xticks(rotation=45)
plt.figure(figsize=(18,6))
nlcd_labels=[]
nlcd_colors=[]
nlcd_tots=np.zeros((len(nlcds['nlcd21'],)),dtype='float')
start = 0
for k in nlcds.keys():
    nlcd_tots=nlcd_tots.copy()+np.array(nlcds[k])
    if k=='nlcd_dom':
        continue
    if start ==0:
        plt.bar(sites,nlcds[k],color=class_colors[int(k[-2:])])
        summ = np.array(nlcds[k])
        start = 1
    else:
        plt.bar(sites,nlcds[k],color=class_colors[int(k[-2:])],bottom=summ)
        summ = summ.copy()+np.array(nlcds[k])
    nlcd_labels.append(class_names[int(k[-2:])])
    nlcd_colors.append(class_colors[int(k[-2:])])
plt.xticks(rotation=45)
plt.legend(nlcd_labels)
print(nlcd_tots)
print(summ)

# %%
nlcd_points={}
nlcd_dom = fp['nlcd_dom'][:]
TSS= fp['TSS_FILTER'][:]
for i in range(len(fpsites)):
    j = nlcd_dom[i]
    if j not in nlcd_points.keys():
        nlcd_points[j]=[]
    nlcd_points[j].append(TSS[i])

# %%
fig,(ax)=plt.subplots(figsize=(10,4))
violin_labels=[]
violin_colors=[]
violin_vectors=[]
nplist=list(nlcd_points.keys())
nplist.sort()
for k in nplist:
    violin_labels.append(class_names[k])
    violin_colors.append(class_colors[k])
    violin_vectors.append(nlcd_points[k])
vio = ax.violinplot(violin_vectors,showmeans=True,showmedians=True,widths=.9)
for k in range(len(vio['bodies'])):
    vio['bodies'][k].set_facecolor(violin_colors[k])
    vio['bodies'][k].set_alpha(.4)
vio['cmeans'].set_color(violin_colors)
vio['cbars'].set_color(violin_colors)
vio['cmaxes'].set_color(violin_colors)
vio['cmins'].set_color(violin_colors)
vio['cmedians'].set_color(violin_colors)
vio['cmedians'].set_linestyle('--')
ax.get_xaxis().set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, len(violin_labels) + 1))
ax.set_xticklabels(violin_labels)
ax.set_xlim(0.25, len(violin_labels) + 0.75)
plt.xticks(rotation=45)
plt.ylabel('$\sigma_T$')
plt.title('Dominant Landcover vs $\sigma_T$')

# %%

# %%
dom_color=[]
slpe=[]
nme=[]
errs=[]
towh=[]
plt.figure(figsize=(15,4))
ordered_sitelist=[]
flip_slope={}
for k in tilts.keys():
    flip_slope[tilts[k]]=k
slpe_list=list(flip_slope.keys())
slpe_list.sort()
for k in slpe_list:
    ordered_sitelist.append(flip_slope[k])

for site in ordered_sitelist:
    dom_color.append(class_colors[site_nlcd_dom[site]])
    slpe.append(tilts[site])
    nme.append(site)
    errs.append(errors[site])
    towh.append(tower_height[site])
bar1 = plt.bar(nme,slpe,color=dom_color)
plt.ylim(.5,1.35)
plt.xticks(rotation=45)

i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, .55,str(int(round(errs[i]*100))), ha='center', va='bottom',color='red')
    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(towh[i]))), ha='center', va='bottom')
    i=i+1

print(site_nlcd_dom.values())
plt.title('Best Fit Slope by Site with Landcover Type: Tower height in Black, % Error in red')
plt.ylabel('Best Fit Slope')

# %%
tilts.values()

# %%
errors.values()

# %%
### THIS IST STATIC NEEDS UPDATING ###
# moisture vs heat errors
heat_errors=np.array([0.11812625474867995, 0.19234469950246175, 0.1965272493547117, 0.11028336129100362, 0.10720607412354952, 0.1320835204144852, 0.1003040438489969, 0.13702979986496935, 0.13188138705917815, 0.10025790106426902, 0.18050863895294342, 0.14094614794369523, 0.18580588032269094, 0.20321937682315433, 0.11585593476450129, 0.13602812289192306, 0.11210168320931511, 0.16315608981948213, 0.10887370958659276, 0.14349675512564625, 0.1248378034144707, 0.1685746481686519, 0.13241921944839863, 0.11549488464298811, 0.14721318395929067, 0.1118688581711944, 0.12137997597675138, 0.11449461673638293, 0.12123487727246669, 0.2237394728610318, 0.2075332220363696, 0.12341634242003362, 0.10575954664910571, 0.14881392135412486, 0.10492596218836962, 0.09192894921644133, 0.12846583372868217, 0.11542302627678856, 0.11563051519680596])
moisture_errors=np.array(list(errors.values()))
print(moisture_errors.shape)
plt.scatter(heat_errors,moisture_errors)

# %%
biases.keys()

# %%
np.where(result>=4)[0][0]

# %%
fp.close()
