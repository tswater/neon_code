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
fp = h5py.File('data/QAQCv4_neon_pntEB0.2_day_rain_frez_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5','r')

# %%
print(fp.keys())

# %%
pred = ['H','USTAR','WS','VEG','TREE','H2O','T_SONIC','BARE_SOIL','LAI','ZL','BLH','TCC','CAPE','SWC']
#pred = ['H','USTAR','WS','H2O','T_SONIC','LAI','ZL','BLH','TCC','CAPE','SWC']
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
result=(fp['TSS_FILTER'][:])

# %%
plt.hist(fp['H2O_SIGMA'][:],bins=100)
plt.title('H2O SIGMA')

# %%
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
#m = m & (fp['H'][:]>0)
#m = m & (isNeon)
m = np.ones((n,),dtype=bool)
m = m&(zL<0)
a = 4
b = 8.3
#c=a*b**(-.66)
#print(c)
#c=.81 #(.69 or 1)
c=.95
delta =0
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
andre_RHS = c**2*(-zL)**(-.66)
andre_RHS[zL>=0]=4
andre_Tsig = np.sqrt(andre_RHS/(ustar**2/H**2))
o_Tsig = result.copy()
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
print(metrics.mean_squared_error(o_Tsig[m]-delta, andre_Tsig[m],squared=False)/np.mean(o_Tsig[m]-delta))
print(np.mean(o_Tsig[m]-andre_Tsig[m]))
print()
uf_Tsig=fp['T_SONIC_SIGMA'][:]
plt.hexbin(fp['T_SONIC_SIGMA'][m]-delta,andre_Tsig[m],gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.5,0,1.5])
plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3)
plt.xlim(0,1.5)
plt.ylim(0,1.5)
plt.title("Equation 1 Validation -- Unfiltered",fontsize=18)
plt.xlabel("Eddy Flux Tower Observed Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
plt.ylabel("EQ 1 Predicted Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
bias_2 = np.sum(andre_Tsig[m]-(fp['T_SONIC_SIGMA'][m]-delta))/len(o_Tsig[m])
print('Bias: '+str(bias_2/np.mean(fp['T_SONIC_SIGMA'][m]-delta)))
print(metrics.mean_squared_error(fp['T_SONIC_SIGMA'][m]-delta, andre_Tsig[m],squared=False)/np.mean(fp['T_SONIC_SIGMA'][m]-delta))
print(np.mean(fp['T_SONIC_SIGMA'][m]-andre_Tsig[m]))

# %%
plt.figure()
plt.hexbin(np.log(zL),ustar**2/H**2*o_Tsig**2,gridsize=300,cmap='terrain',mincnt=1,extent=(-2.5,1.5,0,15))
zL2=np.linspace(-2.5,1.5,250)
zL2=np.log(zL2)
andre_perfect=c**2*(-zL2)**(-.66)
andre_perfect[zL2>0]=a
plt.plot(zL2,andre_perfect,'r--')
plt.xlabel('ZL: Monin-Obukhov Stability Param',fontsize=14)
plt.ylabel('Andre LHS (non-dimensional)',fontsize=14)
plt.legend(['Andre Equation'],fontsize=14,loc='upper left')
plt.title('Monin-Obukhov Stability vs Heat Term',fontsize=18)


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
    model = LinearRegression()
    model.fit(obs.reshape((-1, 1)),pr)
    return model.coef_[0]

def tilt2(obs,pr):
    model = LinearRegression(fit_intercept=False)
    model.fit(obs.reshape((-1, 1)),pr)
    return model.coef_[0]


# %%
plt.figure(figsize=(15,15))
zL = fp['ZL'][:]
i=1
errors={}
biases={}
tower_height={}
tilts={}
canh={}
for site in sites:
    m_site = fpsites[:]==site
    if i == 26:
        i=1
        plt.subplots_adjust(wspace=.3,hspace=.25)
        plt.figure(figsize=(15,15))
    plt.subplot(5,5,i)
    m = np.ones((n,),dtype=bool)
    m = m&(zL<0)
    m=m&m_site
    errors[site]=np.sqrt(metrics.mean_squared_error(o_Tsig[m]-delta, andre_Tsig[m])/np.mean(uf_Tsig[m]-delta))
    biases[site]=np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])/np.mean(o_Tsig[m]-delta)
    tower_height[site]=fp['tow_height'][m][0]
    canh[site]=fp['canopy_height'][m][0]
    plt.hexbin(o_Tsig[m]-delta,andre_Tsig[m],gridsize=50,cmap='terrain',mincnt=1,extent=[0,1.25,0,1.25])
    limmin = np.min([np.min(o_Tsig),np.min(andre_Tsig)])
    limmax = np.max([np.max(o_Tsig),np.max(andre_Tsig)])
    plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3,alpha=.7)
    plt.xlim(0,1.25)
    plt.ylim(0,1.25)
    tilts[site]=tilt(o_Tsig[m],andre_Tsig[m])
    plt.title(str(site)[2:-1]+" "+f'{tilts[site]:1.2f}')
    i=i+1
plt.subplots_adjust(wspace=.3,hspace=.25)

# %%
plt.figure(figsize=(15,15))
i=1
c_s={}
m=fp['ZL'][:]<0
m=m&(fp['H'][:]>=50)
rho = fp['RHO'][m]
zL = fp['ZL'][m]
H = fp['H'][m]/1005/rho
LHS_old=result_old[m]*fp['USTAR'][m]/H
LHS_new=result[m]*fp['USTAR'][m]/H
RHS=(-zL)**(-1/3)
new=True
for site in sites:
    m_site = fpsites[m]==site
    if i == 26:
        i=1
        plt.subplots_adjust(wspace=.3,hspace=.25)
        plt.figure(figsize=(15,15))
    plt.subplot(5,5,i)
    if new:
        min_=np.min(RHS)
        max_=np.max(RHS)
        plt.hexbin(RHS[m_site],LHS_new[m_site],gridsize=50,cmap='terrain',mincnt=1,extent=(0,4,0,4))
        plt.plot([min_,max_],[min_*.95,max_*.95],c='red')
        c_s[site]=tilt2(RHS[m_site],LHS_new[m_site])
        plt.plot([min_,max_],[min_*c_s[site],max_*c_s[site]],c='pink')
    else:
        min_=np.min(RHS)
        max_=np.max(RHS)
        plt.hexbin(RHS[m_site],LHS_old[m_site],gridsize=50,cmap='terrain',mincnt=1,extent=(0,4,0,4))
        plt.plot([min_,max_],[min_*.95,max_*.95],c='red')
        c_s[site]=tilt2(RHS[m_site],LHS_old[m_site])
        plt.plot([min_,max_],[min_*c_s[site],max_*c_s[site]],c='pink')
    plt.title(str(site)[2:-1]+" "+f'{c_s[site]:1.2f}')
    plt.xlim(0,4)
    plt.ylim(0,4)
    i=i+1
plt.subplots_adjust(wspace=.3,hspace=.25)

# %%
i=1
c_s_uf={}
plt.figure(figsize=(15,15))
for site in sites:
    m_site = fpsites[m]==site
    if i == 26:
        i=1
        plt.subplots_adjust(wspace=.3,hspace=.25)
        plt.figure(figsize=(15,15))
    plt.subplot(5,5,i)
    min_=np.min(RHS)
    max_=np.max(RHS)
    plt.hexbin(RHS[m_site],LHS_old[m_site],gridsize=50,cmap='terrain',mincnt=1,extent=(0,4,0,4))
    plt.plot([min_,max_],[min_*.95,max_*.95],c='red')
    c_s_uf[site]=tilt2(RHS[m_site],LHS_old[m_site])
    plt.plot([min_,max_],[min_*c_s[site],max_*c_s[site]],c='pink')
    plt.title(str(site)[2:-1]+" "+f'{c_s[site]:1.2f}')
    plt.xlim(0,4)
    plt.ylim(0,4)
    i=i+1
plt.subplots_adjust(wspace=.3,hspace=.25)

# %%
x=np.linspace(1,4)
y=(x**(1/3))**3
plt.plot(x,y)

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


print(tilts.values())

# %%
len(tower_height.values())

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
### SETUP RF on C1 for OLD and NEW
#pred = ['WS','VEG','TREE','H2O','T_SONIC','BARE_SOIL','LAI','BLH','TCC','CAPE','SWC']
pred = ['WS','H2O','T_SONIC','LAI','BLH','TCC','CAPE','SWC']
m=fp['ZL'][:]<-0.01
m=m&(fp['H'][:]>=50)
n = np.sum(m)
n2 = len(pred)
result_old=(fp['T_SONIC_SIGMA'][m])
result=(fp['TSS_FILTER'][m])
data = np.zeros((n2,n))

i = 0
for x in pred:
    print(x)
    data[i,:]=fp[x][m]
    i = i+1
data = np.transpose(data)
rho = fp['RHO'][m]
zL = fp['ZL'][m]
H = fp['H'][m]/1005/rho
result_old_c=result_old*fp['USTAR'][m]/H/(-zL)**(-1/3)
result_c=result*fp['USTAR'][m]/H/(-zL)**(-1/3)

# %%
X_train, X_test, y_train, y_test, w_train,w_test = train_test_split(data,result_c,weights[m], test_size=0.2, random_state=0)

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
plt.hexbin(y_test,y_pred,cmap='terrain',gridsize=200,mincnt=1)
limmin = np.min([np.min(y_test),np.min(y_pred)])
limmax = np.max([np.max(y_test),np.max(y_pred)])
plt.plot([limmin,limmax],[limmin,limmax],"k:")
#plt.xlim(0,2)
#plt.ylim(0,2)
plt.title("RF Validation -- Filtered")
plt.xlabel("Observed Value Sigma_T (K)")
plt.ylabel("Predicted Value Sigma_T (K)")

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
plt.figure(figsize=(15,15))
i=1
errors={}
biases={}
tower_height={}
for site in sites:
    m_site = fpsites[:]==site
    if i == 26:
        i=1
        plt.figure(figsize=(15,15))
    plt.subplot(5,5,i)
    m=m_site
    errors[site]=np.sqrt(metrics.mean_squared_error(o_Tsig[m]-delta, andre_Tsig[m])/np.mean(o_Tsig[m]-delta))
    biases[site]=np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])/np.mean(o_Tsig[m]-delta)
    tower_height[site]=fp['tow_height'][m][0]
    llab = []
    lcol = []
    lval = []
    for j in range(len(nlcd_colors)):
        if site_nlcds[site][j] >0:
            llab.append(nlcd_labels[j])
            lcol.append(nlcd_colors[j])
            lval.append(site_nlcds[site][j])
    plt.pie(lval,labels=llab,radius=.75,colors=lcol)
    plt.title(str(site)[2:-1])
    i=i+1

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
plt.ylim(.5,1.2)
plt.xticks(rotation=45)

i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, .52,str(int(round(errs[i]*100))), ha='center', va='bottom',color='red')
    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(towh[i]))), ha='center', va='bottom')
    i=i+1

print(site_nlcd_dom.values())
plt.title('Best Fit Slope by Site with Landcover Type: Tower height in Black, % Error in red')
plt.ylabel('Best Fit Slope')

# %%
### RF BY SITE ###
weights=np.array(weights)
error={}
bias={}
fimp={}
names3={}
pred_order={}
colors_rf={}
colors_pred={'H':'tomato','USTAR':'aqua','WS':'skyblue','H2O':'slateblue',
             'T_SONIC':'indianred','LAI':'forestgreen','ZL':'blueviolet',
             'BLH':'gold','TCC':'plum','CAPE':'orange','SWC':'peru'}
#pred = ['H','USTAR','WS','H2O','T_SONIC','LAI','ZL','BLH','TCC','CAPE','SWC']
    
for site in sites:
    print(site)
    m_site=fpsites==site
    data2=np.zeros((np.sum(m_site),data.shape[1]))
    for i in range(data.shape[1]):
        data2[:,i]=data[m_site,i]
    X_train2, X_test2, y_train2, y_test2, w_train2,w_test2 = train_test_split(data2,result[m_site],weights[m_site], test_size=0.2, random_state=0)
    clf = sklearn.ensemble.RandomForestRegressor(random_state=1,n_estimators=100,oob_score=True)
    clf.fit(X_train2,y_train2,w_train2)
    y_pred = clf.predict(X_test2)
    error[site] = np.sqrt(metrics.mean_squared_error(y_test2, y_pred))/np.mean(y_test2)
    bias[site] = np.sum(y_pred-y_test2)/len(y_pred)/np.mean(y_test2)
    #make a names list
    namelist=[]
    colorlist=[]
    predlist=[]
    for x in pred:
        namelist.append(names[x])
        colorlist.append(colors_pred[x])
        predlist.append(x)
    # sort the feature importance
    fIMP = clf.feature_importances_
    points = fIMP
    fIMP_names={}
    fIMP_colors={}
    fIMP_pred={}
    for i in range(len(namelist)):
        fIMP_names[points[i]]=namelist[i]
    for i in range(len(colorlist)):
        fIMP_colors[points[i]]=colorlist[i]
    for i in range(len(colorlist)):
        fIMP_pred[points[i]]=predlist[i]
    points[::-1].sort()
    names2=[]
    colors2=[]
    pred2=[]
    for i in range(len(namelist)):
        colors2.append(fIMP_colors[points[i]])
        names2.append(fIMP_names[points[i]])
        pred2.append(fIMP_pred[points[i]])
    names3[site]=names2
    fimp[site]=points
    colors_rf[site]=colors2
    pred_order[site]=pred2

# %%
### MAKE RESULTS ### 
counts={}
count_colors=[]
for x in pred:
    counts[x]=0
    if x=='H':
        continue
    count_colors.append(colors_pred[x])
i = 1
plt.figure(figsize=(21,25))
for site in sites:
    plt.subplot(7,7,i)
    prednames=[]
    j=0
    for x in pred_order[site]:
        counts[x]=fimp[site][j]+counts[x]
        prednames.append(x[0:4])
        j=j+1
    plt.bar(prednames,fimp[site],color=colors_rf[site])
    #plt.ylim(0,.2)
    #fs=np.sum(fimp[site][1:])
    #plt.pie(fimp[site][1:]/fs,colors=colors_rf[site][1:])
    plt.title(str(site)[2:-1]+'\nerror: '+str(round(error[site]*100))+\
             '\nbias: '+str(round(bias[site]*100))+'\nnlcd: '+class_names[site_nlcd_dom[site]])
    i = i+1
    plt.xticks(rotation=45)
plt.subplots_adjust(wspace=.3,hspace=1.4)
plt.subplot(7,7,i)

# %%
print(errors.values())

# %%
values_cnt=[]
names_cnt=[]
flip_counts={}
flip_count_colors={}
i=0
for k in counts.keys():
    if (k=='H'):
        continue
    flip_counts[counts[k]]=k
    flip_count_colors[counts[k]]=count_colors[i]
    i=i+1
sorted_counts=list(flip_counts.keys())
sorted_counts.sort()
count_colors2=[]
for k in sorted_counts:
    names_cnt.append(flip_counts[k])
    values_cnt.append(k)
    count_colors2.append(colors_pred[flip_counts[k]])
plt.figure()
plt.bar(names_cnt,values_cnt,color=count_colors2)

# %%
len(count_colors)

# %%
print(tilts.values())

# %%
np.where(result>=4)[0][0]

# %%
fp.close()

# %%

# %%
import numpy as np


# 30 points between [0, 0.2) originally made using np.random.rand(30)*.2
pts = np.array([
    0.015, 0.166, 0.133, 0.159, 0.041, 0.024, 0.195, 0.039, 0.161, 0.018,
    0.143, 0.056, 0.125, 0.096, 0.094, 0.051, 0.043, 0.021, 0.138, 0.075,
    0.109, 0.195, 0.050, 0.074, 0.079, 0.155, 0.020, 0.010, 0.061, 0.008])

# Now let's make two outlier points which are far away from everything.
pts[[3, 14]] += .8

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax.plot(pts)
ax2.plot(pts)

# zoom-in / limit the view to different portions of the data
ax.set_ylim(.78, 1.)  # outliers only
ax2.set_ylim(0, .22)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# %%
