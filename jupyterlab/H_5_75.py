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
fp = h5py.File('data/QAQCv4_neon_pntEB0.75_0.5_day_rain_frez_BA_BL_CA_H_H2_H2_H2_LA_SW_TC_TR_T__US_VE_WS_ZL.h5','r')

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
    #plt.title(x+' vs H2O_SIGMA')
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
plt.hexbin(fp['H2O_SIGMA'][m]*18/29-delta,andre_Tsig[m],gridsize=200,cmap='terrain',mincnt=1,extent=[0,1.5,0,1.5])
plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3)
plt.xlim(0,1.5)
plt.ylim(0,1.5)
plt.title("Equation 1 Validation -- Unfiltered",fontsize=18)
plt.xlabel("Eddy Flux Tower Observed Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
plt.ylabel("EQ 1 Predicted Value $\sqrt{\\overline{\\theta'}^2}$ (K)",fontsize=14)
bias_2 = np.sum(andre_Tsig[m]-(fp['H2O_SIGMA'][m]*18/29-delta))/len(o_Tsig[m])
print('Bias: '+str(bias_2/np.mean(fp['H2O_SIGMA'][m]*18/29-delta)))
print(np.sqrt(metrics.mean_squared_error(fp['H2O_SIGMA'][m]*18/29-delta, andre_Tsig[m])/np.mean(fp['H2O_SIGMA'][m]*18/29-delta)))
print(np.mean(fp['H2O_SIGMA'][m]-andre_Tsig[m]))

# %%
plt.hexbin(result,fp['H2O_SIGMA'][:],gridsize=100,cmap='terrain',mincnt=1,extent=[0,1.5,0,1.5])
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
TSS=fp['H2O_SIGMA'][:]

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
    plt.hexbin(o_Tsig[m]-delta,andre_Tsig[m],gridsize=50,cmap='terrain',mincnt=1,extent=[0,.75,0,.75])
    limmin = np.min([np.min(o_Tsig),np.min(andre_Tsig)])
    limmax = np.max([np.max(o_Tsig),np.max(andre_Tsig)])
    plt.plot([limmin,limmax],[limmin,limmax],"k--",linewidth=3,alpha=.7)
    plt.xlim(0,.75)
    plt.ylim(0,.75)
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
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue'}

dom_color=[]
T_2_biases=np.array([0.04550157169712469, 0.2593908065188178, 0.07393466636896899, -0.18042015125503988, -0.03686076713955539, -0.040598130694064975, 0.05931354386000764, -0.21996339176057872, 0.08222710826153021, 0.15197184535264757, -0.2022820923893887, 0.2661054480918105, -0.049764492892687036, 0.03796834510464079, -0.17589690863972565, 0.08795139657642519, 0.0019624827493387184, -0.0815447383747816, -0.023818814385299585, 0.07349724273989684, 0.04766876655619503, 0.024061379081002705, -0.001081221880708063, -0.11743258781288218, 0.018042851828649897, -0.0517362156390338, 0.07030606181049039, 0.258380678682515, 0.0970403042996922, 0.033574698610988815, -0.16066968677118157, 0.06991697192369617, -0.01474167926275867, -0.02098859915198389, -0.33790105272633636, -0.07817664544499328, 0.057862698160667414])
T_2_tilts =np.array([1.0745038840044885, 1.318405581548103, 1.1687848211580973, 0.859292090359501, 0.7872463580235415, 1.0167489760471362, 1.0984892834389364, 0.7466665526111849, 1.078772704532749, 1.2105839463277077, 0.7959430802172207, 1.2600670856886578, 0.9920639056034901, 1.0319425521986834, 0.9282284187407789, 1.0839326403118696, 1.096303113653768, 0.7902734917299935, 1.0269778062557753, 1.1677362127729352, 1.094505939177768, 1.09960467215381, 0.8899971644400713, 0.9749837367273555, 0.9835072624501936, 1.0606618093108653, 1.1171482328085833, 1.1999094582528669, 1.2240345385477929, 1.1115032718960316, 0.9389767998575613, 1.0856368305061126, 0.9576805269840756, 1.0184647832564138, 0.40559731836242513, 0.9228641745186325, 1.2408204904227143])
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
biases.keys()

# %%
np.where(result>=4)[0][0]

# %%
fp.close()
