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
fp = h5py.File('data/QAQCv4_neon_pntEB1_0.5_day_rain_frez_BA_BL_CA_H_H2_LA_SW_TC_TR_TS_T__US_VE_WS_ZL.h5','r')

# %%
print(fp.keys())

# %%
pred = ['H','USTAR','WS','BLH','VEG','TREE','H2O','T_SONIC','ZL','BARE_SOIL','LAI','TCC','CAPE','SWC']
pred = ['H','USTAR','WS','H2O','T_SONIC','LAI','ZL','BLH','TCC','CAPE','SWC']
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
#m = m&(zL>0)
#m = m & (fp['H'][:]>0)
#m = m & (isNeon)
m = np.ones((n,),dtype=bool)
a = 4
b = 8.3
delta =0
ustar = fp['USTAR'][:]
rho = fp['RHO'][:]
H = fp['H'][:]/1005/rho
andre_RHS = a*(1-b*zL)**(-.66)
andre_RHS[zL>=0]=a
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
print(np.sqrt(metrics.mean_squared_error(o_Tsig[m]-delta, andre_Tsig[m])/np.mean(o_Tsig[m]-delta)))
print(np.mean(o_Tsig[m]-andre_Tsig[m]))
print()
plt.hexbin(fp['T_SONIC_SIGMA'][m]-delta,andre_Tsig[m],gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.5,0,1.5])
plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3)
plt.xlim(0,1.5)
plt.ylim(0,1.5)
plt.title("Equation 1 Validation -- Unfiltered",fontsize=18)
plt.xlabel("Eddy Flux Tower Observed Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
plt.ylabel("EQ 1 Predicted Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
bias_2 = np.sum(andre_Tsig[m]-(fp['T_SONIC_SIGMA'][m]-delta))/len(o_Tsig[m])
print('Bias: '+str(bias_2/np.mean(fp['T_SONIC_SIGMA'][m]-delta)))
print(np.sqrt(metrics.mean_squared_error(fp['T_SONIC_SIGMA'][m]-delta, andre_Tsig[m])/np.mean(fp['T_SONIC_SIGMA'][m]-delta)))
print(np.mean(fp['T_SONIC_SIGMA'][m]-andre_Tsig[m]))

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
    model = LinearRegression()
    model.fit(obs.reshape((-1, 1)),pr)
    return model.coef_[0]


# %%
plt.figure(figsize=(15,15))
i=1
errors={}
biases={}
tower_height={}
tilts={}
for site in sites:
    m_site = fpsites[:]==site
    if i == 26:
        i=1
        plt.subplots_adjust(wspace=.3,hspace=.25)
        plt.figure(figsize=(15,15))
    plt.subplot(5,5,i)
    m = np.ones((n,),dtype=bool)
    m=m&m_site
    errors[site]=np.sqrt(metrics.mean_squared_error(o_Tsig[m]-delta, andre_Tsig[m])/np.mean(o_Tsig[m]-delta))
    biases[site]=np.sum(andre_Tsig[m]-(o_Tsig[m]-delta))/len(o_Tsig[m])/np.mean(o_Tsig[m]-delta)
    tower_height[site]=fp['tow_height'][m][0]
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
# PLOT TOW_HEIGHT vs SLOPE
plt.figure()
plt.scatter(tower_height.values(),tilts.values())
plt.xlabel('Tower Height')
plt.ylabel('Slope Best Fit')
# PLOT TOW_HEIGHT vs ERROR
plt.figure()
plt.scatter(tower_height.values(),errors.values())
plt.xlabel('Tower Height')
plt.ylabel('Errors')
# PLOT SLOPE vs ERROR
plt.figure()
plt.scatter(tilts.values(),errors.values())
plt.xlabel('Slope Best Fit')
plt.ylabel('Errors')
# PLOT TOW_HEIGHT vs BIAS
plt.figure()
plt.scatter(tower_height.values(),biases.values())
plt.xlabel('Tower Height')
plt.ylabel('Biases')

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
## THIS IS STATIC WILL NEED TO UPDATE
dom_color=[]
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue'}

T_2_biases=np.array([-0.0337381316121804, 0.1064913340528113, -0.16313965851093878, -0.02957813355194774, 0.025741990015571783, 0.07215290484406182, -0.0772075418077183, 0.027388690396201728, 0.07716945597900654, -0.057323643611187775, 0.05200144110562689, 0.04787555271440314, 0.18066082294416128, 0.14701024478952393, -0.041277991122157874, 0.02050242249995701, -0.021681915085298882, 0.13925162412738665, -0.06797722456062838, 0.08533950016903512, -0.139300279975939, 0.09832393274284852, -0.014862785720739203, -0.11736350186196008, 0.1092320886090514, 0.03436517034821108, 0.036374350502063384, -0.013531826539447778, -0.05537427263231309, -0.07534767356850038, 0.1468980966093707, 0.005920458225999184, -0.030433704760490725, 0.005581931068562691, -0.011179158142269295, -0.0020834672793350775, -0.08780411247956123, -0.016810920052460286, -0.06594295442898059])
T_2_tilts =np.array([1.1084314001713185, 1.3109002249689545, 0.9380524508234767, 1.0952747402270897, 1.116404303282227, 1.2611364274150427, 1.0721968764127219, 1.2211941699276083, 1.2013402651390965, 1.0408256548476134, 1.3185074185779995, 1.1608404005047803, 1.2759822082821113, 1.2680792239632834, 1.0783651559565468, 1.1587092123710714, 1.0938403620134354, 1.266718899633263, 1.0957139802303508, 1.0718458315481523, 0.9896556192288103, 1.2452569138867662, 1.1623506129638599, 1.0596371552892776, 1.193496427164617, 1.1021482965610396, 1.158931580425429, 1.133002198278727, 1.1458494830023551, 1.1118177843406134, 1.3289548836795904, 1.1992133383956185, 1.065994160689183, 1.1918135540239922, 1.0471538520497594, 1.102093073573164, 1.0228709954548882, 1.1045751051392414, 1.047871450052403])
T_5_biases=np.array(list(biases.values()))
T_5_tilts = np.array(list(tilts.values()))
delta_tilts=-T_2_tilts+T_5_tilts
delta_biases=-T_2_biases+T_5_biases
for site in sites:
    dom_color.append(class_colors[site_nlcd_dom[site]])
plt.figure(figsize=(15,4))
plt.bar(sites,delta_tilts,color=dom_color)
plt.xticks(rotation=45)
plt.title('Change in Best Fit Slope from Balanced EB to Unbalaced EB')
plt.figure(figsize=(15,4))
plt.bar(sites,delta_biases,color=dom_color)
plt.xticks(rotation=45)
plt.title('Change in Bias from Balanced EB to Unbalaced EB')

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
plt.ylim(.6,1.25)
plt.xticks(rotation=45)

i=0
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, .62,str(int(round(errs[i]*100))), ha='center', va='bottom',color='red')
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
             'BLH':'gold','TCC':'plum','CAPE':'orange','SWC':'peru',
             'VEG':'','TREE':'','BARE_SOIL':''}
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Frst',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wood Wet',95:'Herb Wet'}

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
np.where(result>=4)[0][0]

# %%
fp.close()
