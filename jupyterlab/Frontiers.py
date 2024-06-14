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
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.ensemble
from sklearn import metrics
import seaborn as sns
sns.set_theme()

# %%
fp2 = h5py.File('data/QAQC_0.25.h5','r')
fp5 = h5py.File('data/QAQC_0.5.h5','r')
fp1 = h5py.File('data/QAQC_1.h5','r')

# %%
fp1.close()


# %%
def getEB(fp):
    EB_ = fp['NETRAD'][:]-fp['H'][:]-fp['LE'][:]-fp['G'][:]
    EB_pct_= EB_/fp['NETRAD'][:]
    return EB_,EB_pct_


# %%
def getangle(fp,dx):
    u=fp['U'][:]
    v=fp['V'][:]
    xgrad=fp['LST_'+str(dx)+'_XGRAD'][:]
    ygrad=fp['LST_'+str(dx)+'_YGRAD'][:]
    xgrada=fp['LST_'+str(dx)+'_XGRAD_ABS'][:]
    ygrada=fp['LST_'+str(dx)+'_YGRAD_ABS'][:]
    xgradn=xgrad/np.sqrt(xgrad**2+ygrad**2)
    ygradn=ygrad/np.sqrt(ygrad**2+xgrad**2)
    un=u/np.sqrt(u**2+v**2)
    vn=v/np.sqrt(u**2+v**2)
    dot=[]
    for i in range(len(u)):
        dot.append(np.dot([u[i],v[i]],[xgradn[i],ygradn[i]]))
    alpha= np.arccos(dot)/(np.sqrt(un**2+vn**2)+np.sqrt(xgradn**2+ygradn**2))
    alpha[alpha>np.pi]=float('nan')
    alpha=np.degrees(alpha)
    alpha[alpha>90]=180-alpha[alpha>90]
    
    # now with abs
    dot=[]
    for i in range(len(u)):
        dot.append(np.dot([u[i],v[i]],[xgrada[i],ygrada[i]]))
    alphaa= np.arccos(dot)/(np.sqrt(u**2+v**2)+np.sqrt(xgrada**2+ygrada**2))
    alphaa[alphaa>np.pi]=float('nan')
    alphaa=np.degrees(alphaa)
    alphaa[alphaa>90]=180-alphaa[alphaa>90]
    return alpha,alphaa


# %%
alph1,alph1a = getangle(fp1,1)
alph2,alph2a = getangle(fp2,0.25)
alph5,alph5a = getangle(fp5,0.5)

# %%
plt.figure(figsize=(7,3.5))
EB_=getEB(fp1)[1]
m=(fp1['qH2O'][:]==0)
m=m&(EB_<1)
m=m&(EB_>-.4)
#m=m&(np.abs(EB_)<1000)
#m=m&(doy>100)&(doy<200)
m=m&(fp1['NETRAD'][:]>300)
m=m&(fp1['ZL'][:]<0)
EB_plt=EB_[m]
plt.hist(EB_plt*100,bins=100,density=True,histtype='stepfilled',alpha=.65)
plt.xlabel('Percentage EB Residual',fontsize=25)
plt.ylabel('Frequency',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# %%
EB_=getEB(fp1)[0]
m=(fp1['qH2O'][:]==0)
#m=m&(EB_<1)
#m=m&(EB_>-.4)
m=m&(np.abs(EB_)<1000)
#m=m&(doy>100)&(doy<200)
m=m&(fp1['NETRAD'][:]>300)
m=m&(fp1['ZL'][:]<0)
EB_plt=EB_[m]

# %%
m2=m&(hod>0)
ebh=np.zeros((int(np.max(hod[m2]*2))+1,))
cnth=np.zeros((int(np.max(hod[m2]*2))+1,))
for i in range(len(hod[m2])):
    h=hod[m2][i]
    hdx=int(h*2)
    ebh[hdx]=ebh[hdx]+EB_[m2][i]
    cnth[hdx]=cnth[hdx]+1

# %%
EB_out=ebh/(cnth+1)
EB_out[cnth<10]=float('nan')

# %%

# %%
a=np.linspace(0,np.max(hod[m2]),int(np.max(hod[m2]))*2+1)

# %%
plt.figure(figsize=(7,3))
plt.plot(a[0:-1:2],EB_out[0:-1:2],linewidth=3)
plt.xlabel('Hour of Day',fontsize=25)
plt.ylabel('EB Residual \n(W $m^{-2}$)',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# %%
std=fp1['LST_1_STD'][:]#/(fp1['LST_1_MEAN'][:]-273)

# %%
plt.scatter(doy[m2],EB_[m2],alpha=.01)

# %%
plt.hist(utcoff)


# %%
def plot_xEB(x,EB_,a_,w_,alim=0,wlim=9999):
    m=~np.isnan(a_)&~np.isnan(x)
    m=m&(a_>alim)
    m=m&(w_<wlim)
    m=m&(EB_<500)
    x=x[m]
    EB_=EB_[m]
    plt.hexbin(x,EB_,mincnt=1,cmap='terrain',gridsize=40,extent=(np.percentile(x,0.5),np.percentile(x,99.5),np.percentile(EB_,0.5),np.percentile(EB_,99.5)))
    return stats.spearmanr(x,EB_)[0]


# %%
m=fp1['site'][:]==b'SRER'
m=m&(fp1['ZL'][:]<0)
plt.scatter(fp1['LST_1_STD'][:][m],np.log(-fp1['ZL'][:][m]),alpha=.05)
#plt.xscale('log')
r=stats.spearmanr(np.log(-fp1['ZL'][:][m]),fp1['LST_1_STD'][:][m])
print(r)

# %%
from mpl_toolkits import mplot3d

# %%
std=fp1['LST_1_STD'][:]
grad=np.sqrt(fp1['LST_1_XGRAD'][:]**2+fp1['LST_1_YGRAD'][:]**2)
plot_xEB(fp1['USTAR'][:],(getEB(fp1)[1]),alph1,fp1['WS'][:],0)

# %%
fig = plt.figure()
m=fpsites==b'SRER'
ax = fig.add_subplot(projection='3d')
ax.scatter(fp1['USTAR'][m],fp1['LST_1_STD'][m],getEB(fp1)[1][m],alpha=0.02)
plt.xlim(0,1)


# %%
import datetime

# %%
doy=[]
hod=[]
lstd=np.zeros((len(fpsites),))
dt= datetime.datetime(1970,1,1,0,0)
dayold=dt.date()
siteold='asdf'
times=fp1['TIME'][:]
stds=fp1['LST_1_STD'][:]
ilist=[]
lsts=[]
utcoff=fp1['utc_off'][:]
for i in range(len(fp1['TIME'][:])):
    t=times[i]
    dtt=dt+datetime.timedelta(seconds=t)
    doy.append(dtt.timetuple().tm_yday)
    hod.append(dtt.hour+utcoff[i])
    if (siteold!=fpsites[i]) or (dayold!=dtt.date()):
        lstd[ilist]=np.mean(lsts)
        lsts=[]
        ilist=[]
    ilist.append(i)
    lsts.append(stds[i])
    dayold=dtt.date()
    siteold=fpsites[i]
doy=np.array(doy)
hod=np.array(hod)
hod[hod<=0]=24+hod[hod<=0]

# %%
hod[hod<=0]=24+hod[hod<=0]

# %%

# %%
import matplotlib as mpl
from sklearn import linear_model

# %%
plt.hist(fp1['EBDp'][:])

# %%

rs=[]
sites2=[b'SRER']
std=fp1['LST_1_STD'][:]#/(fp1['LST_1_MEAN'][:]-273)
for site in sites2:
    fig,ax=plt.subplots(figsize=(8,4))
    m=fpsites==site
    m=m&(fp1['NETRAD'][:]>300)
    m=m&(fp1['G'][:]<200)
    m=m&(fp1['ZL'][:]<0)
    m=m&(bowen>0)&(bowen<20)
    m=m&(fp1['qH2O'][:]==0)
    EB_=getEB(fp1)[1][m]*100
    color=plt.cm.jet((EB_-np.percentile(EB_,1))/(np.percentile(EB_,99)-np.percentile(EB_,1)))
    #ustr=np.array(doy)[m]
    ustr=fp1['T_SONIC_SIGMA'][m]/fp1['H'][m]*100
    lstr=std[m]#fp1['LST_1_STD'][m]/(fp1['LST_1_MEAN'][m]-273)
    plt.scatter(ustr,lstr,color=color,s=9,alpha=.7)
    plt.xlim(np.percentile(ustr,.1),np.percentile(ustr,99.5))
    plt.ylim(np.percentile(lstr,.1),np.percentile(lstr,99.5))
    plt.xlabel('$\sigma_{Tair}/H$ x 100',fontsize=26)
    plt.ylabel('$\sigma_{LST}$',fontsize=26)
    try:
        regr = linear_model.LinearRegression()
        regr.fit(np.array([ustr,lstr]).T, EB_)
        scr=regr.score(np.array([ustr,lstr]).T, EB_)
    except:
        scr=0
    rs.append(scr)
    #plt.title(str(site)+' '+str(scr))
    plt.xticks(fontsize=20,rotation=45)
    plt.yticks(fontsize=20)
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=np.percentile(EB_,1), vmax=np.percentile(EB_,99))
    cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label('% EB Res.',fontsize=26)
    cbar.ax.tick_params(labelsize=22)

# %%
plt.hist(rs)

# %%
print(np.mean(rs))

# %%
sites=[]
fpsites=fp1['site'][:]
for site in fpsites:
    if site not in sites:
        sites.append(site)
sites.sort()
sites=np.array(sites)


# %%
def plot_xEB(x,EB_,a_,w_,alim=0,wlim=9999):
    m=~np.isnan(a_)&~np.isnan(x)
    m=m&(a_>alim)
    m=m&(w_<wlim)
    m=m&(EB_<500)
    x=x[m]
    EB_=EB_[m]
    plt.hexbin(x,EB_,mincnt=1,cmap='terrain',gridsize=40,extent=(np.percentile(x,0.5),np.percentile(x,99.5),np.percentile(EB_,0.5),np.percentile(EB_,99.5)))
    return stats.spearmanr(x,EB_)[0]


# %%
rs=[]
bowen=fp1['H'][:]/fp1['LE'][:]
std=fp1['LST_1_STD'][:]/(fp1['LST_1_MEAN'][:]-273)
sites2=[b'SRER']
for site in sites2:
    plt.figure()
    m=fp1sites==site
    m=m&(bowen>0)&(bowen<20)
    m=m&(fp1['qH2O'][:]==0)
    #m=m&(doy>100)&(doy<200)
    m=m&(fp1['NETRAD'][:]>300)
    m=m&(fp1['ZL'][:]<0)
    m=m&(hod==16)
    EB=(getEB(fp1)[1])[m]#/(fp1['H'][m]+fp1['LE'][m])
    #var=fp1['T_SONIC_SIGMA'][m]/fp1['H'][m]#np.log(-fp1['ZL'][m])#fp1['WS'][m]
    #var=fp1['H'][m]
    var=std[m]
    r=plot_xEB(var,EB,alph1[m],fp1['WS'][:][m],0)
    rs.append(r)
    plt.title(str(site)+' '+str(r))
plt.figure()
plt.hist(rs)

# %%
np.mean(rs)

# %%
plt.hist(rs,bins=[0,.1,.2,.3,.4,.5,.6,.7,.8])

# %%
plt.figure(figsize=(14,4))
plt.bar(sites,rs)
plt.xticks(rotation=45)
plt.title('TSSH')

# %%
pred=['CD','H2O','TA','WS','ZL','NETRAD','USTAR','ANGLE','BOWEN','LST_1_STD']
pred=['H2O','TA','NETRAD','BOWEN','LST_1_MEAN','WS','USTAR','T_SONIC_SIGMA','LST_1_STD']
bowen=fp1['H'][:]/fp1['LE'][:]
m=~np.isnan(alph1)
m=m&(fp1['ZL'][:]<0)
m=m&~np.isnan(bowen)
m=m&(bowen<50)
m=m&(bowen>0)
n = len(fp1['H'][m])
n2 = len(pred)
result=getEB(fp1)[1][m]

# %%
data=np.zeros((n2,n))
i = 0
fig = plt.figure(figsize=(12,9))
for x in pred:
    if x == 'ANGLE':
        data[i,:]=alph1[m]
    elif x == 'BOWEN':
        data[i,:]=bowen[m]
    elif x == 'GRAD':
        data[i,:]=grad[m]
    else:
        data[i,:]=fp1[x][m]
    plt.subplot(3,4,i+1)
    try:
        plt.hexbin(data[i,:],result,gridsize=150,cmap='terrain',mincnt=1,extent=(mins[x],maxs[x],rmin,rmax))
    except Exception:
        plt.hexbin(data[i,:],result,gridsize=150,cmap='terrain',mincnt=1)
    plt.title(x,fontsize=12)
    #plt.xlabel(xlabels[x],fontsize=10)
    #plt.title(names[x]+' '+xlabels[x],fontsize=12)
    i = i+1
#fig.suptitle('Direct Comparison of Environmental Covariates vs $\sigma_T$',fontsize=20)
data = np.transpose(data)

# %%
X_train, X_test, y_train, y_test = train_test_split(data,result, test_size=0.2, random_state=0)

# Run RF
clf = sklearn.ensemble.RandomForestRegressor(random_state=1,n_estimators=100,oob_score=True)
clf.fit(X_train,y_train)

# %%
y_pred = clf.predict(X_test)
plt.figure()
minn=min(np.min(y_test2),np.min(y_pred))
minn=-1
maxx=max(np.max(y_test2),np.max(y_pred))
plt.hexbin(y_test,y_pred,cmap='terrain',gridsize=200,mincnt=1,extent=(minn,maxx,minn,maxx))
plt.plot([minn,maxx],[minn,maxx],'--',color='grey',alpha=.4)

# %%
print(clf.oob_score_)

# %%
fIMP = clf.feature_importances_
plt.bar(pred,fIMP)
#plt.title("RF Feature Importance")
#plt.xlabel("Predictor")
plt.ylabel("Overall Feature Importance")
plt.xticks(rotation=45,fontsize=12)
#plt.grid(False,axis='x')
bias = np.sum(y_pred-y_test)/len(y_pred)
print('Bias: '+str(bias/np.mean(y_test)))
print('Normalized RMSE: '+str(metrics.mean_squared_error(y_test, y_pred,squared=False)/np.mean(y_test)))
print('OOB Score: '+ str(clf.oob_score_))

# %%
grad=np.sqrt(fp1['LST_1_XGRAD'][:]**2+fp1['LST_1_YGRAD'][:]**2)
pred=['H2O','TA','NETRAD','BOWEN','LST_1_MEAN','WS','USTAR','T_SONIC_SIGMA','LST_1_STD']
fpsites=fp1['site'][:]

# %%
#### SITE BY SITE RF ####
import warnings
warnings.filterwarnings('ignore')
fimps={}
oobs={}
for site in sites:
    print(str(site)+'...',end='')
    m=fp1['site'][:]==site
    m=m&~np.isnan(alph1)
    m=m&(fp1['ZL'][:]<0)
    m=m&~np.isnan(bowen)
    m=m&(bowen<50)
    m=m&(bowen>0)
    n = len(fp1['H'][m])
    n2 = len(pred)
    result=getEB(fp1)[1][m]
    data=np.zeros((n2,n))
    i = 0
    for x in pred:
        if x == 'ANGLE':
            data[i,:]=alph1[m]
        elif x == 'BOWEN':
            data[i,:]=bowen[m]
        elif x == 'GRAD':
            data[i,:]=grad[m]
        else:
            data[i,:]=fp1[x][m]
        i=i+1
    #fig.suptitle('Direct Comparison of Environmental Covariates vs $\sigma_T$',fontsize=20)
    data = np.transpose(data)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data,result, test_size=0.2, random_state=0)

    # Run RF
    clf2 = sklearn.ensemble.RandomForestRegressor(random_state=1,n_estimators=100,oob_score=True)
    print('setup...'+str(len(y_train2))+'...',flush=True,end='')
    clf2.fit(X_train2,y_train2)
    print('fitted...',flush=True,end='')
    y_pred = clf2.predict(X_test2)
    print('predictted...',flush=True,end='')
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    minn=min(np.min(y_test2),np.min(y_pred))
    maxx=max(np.max(y_test2),np.max(y_pred))
    plt.hexbin(y_test2,y_pred,cmap='terrain',gridsize=50,mincnt=1,extent=(minn,maxx,minn,maxx))
    plt.title(site)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.plot([minn,maxx],[minn,maxx],'--',color='grey',alpha=.4)
    fIMP = clf2.feature_importances_
    fimps[site]=fIMP
    plt.subplot(1,2,2)
    plt.bar(pred,fIMP)
    oobs[site]=clf2.oob_score_
    plt.title("RF Feature Importance "+str(clf2.oob_score_))
    plt.xticks(rotation=45,fontsize=12)
    print('plotted',flush=True)

# %%
np.mean(np.array(list(oobs.values())))

# %%
plot_values=np.zeros((len(pred),len(sites)))
for i in range(len(pred)):
    for j in range(len(sites)):
        plot_values[i,j]=fimps[sites[j]][i]
plt.figure(figsize=(6,4))
ax=sns.violinplot(data=plot_values.transpose(),scale='width',linewidth=.75)
ax.set_xticklabels(pred)
plt.xticks(rotation=45,fontsize=12)

# %%
goodsite=[]
for site in sites:
    if oobs[site]<.5:
        continue
    else:
        goodsite.append(site)
plot_values=np.zeros((len(pred),len(goodsite)))
for i in range(len(pred)):
    for j in range(len(goodsite)):
        plot_values[i,j]=fimps[goodsite[j]][i]
plt.figure(figsize=(6,4))
ax=sns.violinplot(data=plot_values.transpose(),scale='width',linewidth=.75)
ax.set_xticklabels(pred)
plt.xticks(rotation=45,fontsize=12)

# %%

# %%
print(len(goodsite))

# %%
m=~np.isnan(alph1)
m=m&(fp1['ZL'][:]<0)
m=m&~np.isnan(bowen)
m=m&(fp1['NETRAD'][:]>300)
m=m&(bowen<50)
m=m&(bowen>0)
m=m&~np.isnan(fp1['LST_1_CL20'][:])
plt.scatter(fp1['T_SONIC_SIGMA'][m]/fp1['H'][m],fp1['H'][m],s=1,alpha=.05)
plt.xlim(0,.01)
plt.ylim(0,600)

# %%
#### CORRELATION MATRIX ####
vars_cm=['HOD','TA','T_SONIC_SIGMA','DELT','RH','logZL','H','LE','BOWEN','G','NETRAD','WS','USTAR','LST_1_STD','COV','LST_1_CL20','EBp','EB']
vars_nm=['Hour','$T_{air}$','$\sigma_{Tair}$','$LST-T_{air}$','RH','Stability','H','LE','H/LE','G','Net Rad.','Wind','$u_*$','$\sigma_{LST}$','$\sigma_{LST}/\mu_{LST}$','$L_{het LST}$','% EB Res.','EB Res']
bowen=fp1['H'][:]/fp1['LE'][:]
m=~np.isnan(alph1)
m=m&(fp1['ZL'][:]<0)
m=m&~np.isnan(bowen)
m=m&(fp1['NETRAD'][:]>300)
m=m&(bowen<50)
m=m&(bowen>0)
m=m&(hod==16)
m=m&~np.isnan(fp1['LST_1_CL20'][:])
datacm=np.zeros((len(vars_cm),len(vars_cm)))
for v in range(len(vars_cm)):
    var1=vars_cm[v]
    if var1 == 'ANGLE':
        dvar1=alph1[m]
    elif var1=='TSSH':
        dvar1=fp1['T_SONIC_SIGMA'][m]/fp1['H'][m]
    elif var1=='COV':
        dvar1=fp1['LST_1_STD'][m]/(fp1['LST_1_MEAN'][m]-273)
    elif var1 == 'BOWEN':
        dvar1=bowen[m]
    elif var1 == 'GRAD':
        dvar1=grad[m]
    elif var1 == 'logZL':
        dvar1=np.log(-fp1['ZL'][m])
    elif var1 == 'EBp':
        dvar1=getEB(fp1)[1][m]
    elif var1 == 'EB':
        dvar1=getEB(fp1)[0][m]
    elif var1 == 'DOY':
        dvar1=doy[m]
    elif var1 == 'DELT':
        dvar1=fp1['LST_1_MEAN'][m]-(fp1['TA'][m]+273)
    elif var1 == 'HOD':
        dvar1=hod[m]
    else:
        dvar1=fp1[var1][m]
    for v2 in range(len(vars_cm)):
        var2=vars_cm[v2]
        if var2 == 'ANGLE':
            dvar2=alph1[m]
        elif var2=='COV':
            dvar2=fp1['LST_1_STD'][m]/(fp1['LST_1_MEAN'][m]-273)
        elif var2 == 'BOWEN':
            dvar2=bowen[m]
        elif var2 == 'GRAD':
            dvar2=grad[m]
        elif var2 == 'logZL':
            dvar2=np.log(-fp1['ZL'][m])
        elif var2== 'EBp':
            dvar2=getEB(fp1)[1][m]
        elif var2 == 'EB':
            dvar2=getEB(fp1)[0][m]
        elif var2 == 'DOY':
            dvar2=doy[m]
        elif var2=='TSSH':
            dvar2=fp1['T_SONIC_SIGMA'][m]/fp1['H'][m]
        elif var2 == 'DELT':
            dvar2=fp1['LST_1_MEAN'][m]-(fp1['TA'][m]+273)
        elif var2 == 'HOD':
            dvar2=hod[m]
        else:
            dvar2=fp1[var2][m]
        
        datacm[v,v2]=stats.spearmanr(dvar1,dvar2)[0]
fig, ax = plt.subplots(figsize=(8,9))
#datacm[np.abs(datacm)>=.9999]=float('nan')
im = ax.imshow(datacm,cmap='bwr',vmin=-.8,vmax=.8)
#im = ax.imshow(np.abs(datacm),cmap='jet',vmin=0,vmax=1)
plt.colorbar(im)
plt.grid(False)
ax.set_yticks(np.arange(len(vars_cm)))
ax.set_yticklabels(vars_nm)
ax.set_xticks(np.arange(len(vars_cm)))
ax.set_xticklabels(vars_nm)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
print()

# %%
fig, ax = plt.subplots(figsize=(12,12))
#datacm[np.abs(datacm)>=.9999]=float('nan')
im = ax.imshow(datacm,cmap='coolwarm',vmin=-.8,vmax=.8)
#im = ax.imshow(np.abs(datacm),cmap='jet',vmin=0,vmax=1)
cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
cbar.set_label('Correlation', rotation=270,fontsize=30)
cbar.ax.tick_params(labelsize=20)
plt.grid(False)
ax.set_yticks(np.arange(len(vars_cm)))
ax.set_yticklabels(vars_nm,fontsize=25)
ax.set_xticks(np.arange(len(vars_cm)))
ax.set_xticklabels(vars_nm,fontsize=25)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
print()

# %%

# %%
#### CORRELATION MATRIX ####
vars_cm=['HOD','TA','T_SONIC_SIGMA','DELT','RH','logZL','H','LE','BOWEN','G','NETRAD','WS','USTAR','LST_1_STD','COV','LST_1_CL20','EBp','EB']
cms_std=np.zeros((len(sites),len(vars_cm)))
cms_ebp=np.zeros((len(sites),len(vars_cm)))
cms_cov=np.zeros((len(sites),len(vars_cm)))
cms_tssh=np.zeros((len(sites),len(vars_cm)))
all_cms={}
i=0
for site in sites:
    print(site)
    #vars_cm=['CD','COV','T_SONIC_SIGMA','H2O','DOY','DELT','H','LE','G','TA','WS','logZL','ZL','NETRAD','USTAR','ANGLE','BOWEN','LST_1_STD','LST_1_CL20','LST_1_MEAN','GRAD','EBp','EB']
    bowen=fp1['H'][:]/fp1['LE'][:]
    m=~np.isnan(alph1)
    m=m&(fp1['ZL'][:]<0)
    m=m&~np.isnan(bowen)
    m=m&(fp1['NETRAD'][:]>250)
    m=m&(fp1['qH2O'][:]==0)
    m=m&(fpsites==site)
    datacm=np.zeros((len(vars_cm),len(vars_cm)))
    for v in range(len(vars_cm)):
        var1=vars_cm[v]
        if var1 == 'ANGLE':
            dvar1=alph1[m]
        elif var1=='TSSH':
            dvar1=fp1['T_SONIC_SIGMA'][m]/fp1['H'][m]
        elif var1=='COV':
            dvar1=fp1['LST_1_STD'][m]/(fp1['LST_1_MEAN'][m]-273)
        elif var1 == 'BOWEN':
            dvar1=bowen[m]
        elif var1 == 'GRAD':
            dvar1=grad[m]
        elif var1 == 'logZL':
            dvar1=np.log(-fp1['ZL'][m])
        elif var1 == 'EBp':
            dvar1=getEB(fp1)[1][m]
        elif var1 == 'EB':
            dvar1=getEB(fp1)[0][m]
        elif var1 == 'DOY':
            dvar1=doy[m]
        elif var1 == 'DELT':
            dvar1=fp1['LST_1_MEAN'][m]-(fp1['TA'][m]+273)
        elif var1 == 'HOD':
            dvar1=hod[m]
        else:
            dvar1=fp1[var1][m]
        for v2 in range(len(vars_cm)):
            var2=vars_cm[v2]
            if var2 == 'ANGLE':
                dvar2=alph1[m]
            elif var2=='COV':
                dvar2=fp1['LST_1_STD'][m]/(fp1['LST_1_MEAN'][m]-273)
            elif var2 == 'BOWEN':
                dvar2=bowen[m]
            elif var2 == 'GRAD':
                dvar2=grad[m]
            elif var2 == 'logZL':
                dvar2=np.log(-fp1['ZL'][m])
            elif var2== 'EBp':
                dvar2=getEB(fp1)[1][m]
            elif var2 == 'EB':
                dvar2=getEB(fp1)[0][m]
            elif var2 == 'DOY':
                dvar2=doy[m]
            elif var2=='TSSH':
                dvar2=fp1['T_SONIC_SIGMA'][m]/fp1['H'][m]
            elif var2 == 'DELT':
                dvar2=fp1['LST_1_MEAN'][m]-(fp1['TA'][m]+273)
            elif var2 == 'HOD':
                dvar2=hod[m]
            else:
                dvar2=fp1[var2][m]
            datacm[v,v2]=stats.spearmanr(dvar1,dvar2)[0]
            if var1=='EBp':
                cms_ebp[i,v2]=datacm[v,v2]
            if var1=='LST_1_STD':
                cms_std[i,v2]=datacm[v,v2]
            if var1=='COV':
                cms_cov[i,v2]=datacm[v,v2]
            if var1=='TSSH':
                cms_tssh[i,v2]=datacm[v,v2]
    #fig, ax = plt.subplots(figsize=(8,9))
    #im = ax.imshow(datacm,cmap='bwr',vmin=-1,vmax=1)
    #im = ax.imshow(np.abs(datacm),cmap='jet',vmin=0,vmax=1)
    #plt.colorbar(im)
    #plt.grid(False)
    #ax.set_yticks(np.arange(len(vars_cm)))
    #ax.set_yticklabels(vars_cm)
    #ax.set_xticks(np.arange(len(vars_cm)))
    #ax.set_xticklabels(vars_cm)
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #     rotation_mode="anchor")
    #plt.title(site)
    all_cms[site]=datacm.copy()
    i=i+1

# %%
a=fp1['LST_1_CL20'][:]

# %%
np.sum(np.isnan(a))/len(a)

# %%
fig, ax = plt.subplots(figsize=(14,14))
im = ax.imshow(cms_ebp,cmap='bwr',vmin=-.8,vmax=.8)
#im = ax.imshow(cms_ebp,cmap='jet')
plt.colorbar(im)
plt.grid(False)
ax.set_yticks(np.arange(len(sites)))
ax.set_yticklabels(sites)
ax.set_xticks(np.arange(len(vars_cm)))
ax.set_xticklabels(vars_cm)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
plt.title('EBP',fontsize=16)
i=i+1

# %%
fig,ax=plt.subplots(figsize=(8,3))
vars_nm2=[]
cms_ebp2=[]
for i in range(len(vars_nm)):
    if 'EB' in vars_nm[i]:
        continue
    else:
        vars_nm2.append(vars_nm[i])
        cms_ebp2.append(cms_ebp[-11,i])
plt.bar(vars_nm2,cms_ebp2,color='darkred')
plt.xticks(rotation=55,fontsize=20)
plt.setp(ax.get_xticklabels(), ha="right",
         rotation_mode="anchor")
plt.yticks(fontsize=22)
#plt.title('hello')
plt.ylabel('Correlation',fontsize=25)

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,cms_std[-11,:])
plt.xticks(rotation=45)
plt.title('hello')

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,cms_cov[-11,:])
plt.xticks(rotation=45)
plt.title('hello')

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(np.abs(cms_ebp),axis=0))
plt.xticks(rotation=45)
plt.title('hello')

# %%
plt.scatter(cms_ebp[:,3],cms_ebp[:,6])
plt.plot([0,.7],[0,-.7],'--',alpha=.3)

# %%
plt.scatter(cms_ebp[:,2],cms_ebp[:,6])

# %%
fig, ax = plt.subplots(figsize=(14,14))
#im = ax.imshow(datacm,cmap='bwr',vmin=-1,vmax=1)
#im = ax.imshow((cms_std),cmap='jet',vmin=-1,vmax=1)
im = ax.imshow(cms_std,cmap='bwr',vmin=-.8,vmax=.8)
plt.colorbar(im)
plt.grid(False)
ax.set_yticks(np.arange(len(sites)))
ax.set_yticklabels(sites)
ax.set_xticks(np.arange(len(vars_cm)))
ax.set_xticklabels(vars_cm)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
plt.title('LST STD',fontsize=16)
i=i+1

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,cms_std[-11,:])
plt.xticks(rotation=45)
plt.title('hello')

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(np.abs(cms_std),axis=0))
plt.xticks(rotation=45)
plt.title('hello')

# %%
fig, ax = plt.subplots(figsize=(14,14))
#im = ax.imshow(datacm,cmap='bwr',vmin=-1,vmax=1)
#im = ax.imshow(cms_cov,cmap='jet')
im = ax.imshow(cms_cov,cmap='bwr',vmin=-.8,vmax=.8)
plt.colorbar(im)
plt.grid(False)
ax.set_yticks(np.arange(len(sites)))
ax.set_yticklabels(sites)
ax.set_xticks(np.arange(len(vars_cm)))
ax.set_xticklabels(vars_cm)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
plt.title('COV',fontsize=16)
i=i+1

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(cms_cov,axis=0))
plt.xticks(rotation=45)
plt.title('hello')

# %%

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(np.abs(cms_cov),axis=0))
plt.xticks(rotation=45)
plt.title('hello')

# %%
plt.hist(cms_cov[:,-1])

# %%
fig, ax = plt.subplots(figsize=(14,14))
#im = ax.imshow(datacm,cmap='bwr',vmin=-1,vmax=1)
#im = ax.imshow(cms_cov,cmap='jet')
im = ax.imshow(cms_tssh,cmap='bwr',vmin=-.8,vmax=.8)
plt.colorbar(im)
plt.grid(False)
ax.set_yticks(np.arange(len(sites)))
ax.set_yticklabels(sites)
ax.set_xticks(np.arange(len(vars_cm)))
ax.set_xticklabels(vars_cm)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
plt.title('TSSH',fontsize=16)
i=i+1

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(cms_tssh,axis=0))
plt.xticks(rotation=45)
plt.title('hello')

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(np.abs(cms_tssh),axis=0))
plt.xticks(rotation=45)
plt.title('hello')

# %%

# %%

# %%
plt.figure(figsize=(5,5))
plt.scatter(cms_std[:,1],cms_std[:,-2])
plt.xlim(-.4,.7)
plt.ylim(-.7,.4)

# %%
vars_cm=['CD','T_SONIC_SIGMA','H2O','DOY','DELT','H','LE','G','TA','WS','logZL','ZL','NETRAD','USTAR','ANGLE','BOWEN','LST_1_STD','LST_1_CL20','LST_1_MEAN','GRAD','EBp','EB']


# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(cms_ebp,axis=0))
plt.xticks(rotation=45)
plt.title('hello')
plt.ylim(-.6,.2)

# %%
plt.figure(figsize=(12,6))
plt.bar(vars_cm,np.mean(cms_std,axis=0))
plt.xticks(rotation=45)
plt.title('hello')
plt.ylim(-.2,.35)

# %%
fp1.close()

# %%
np.mean(fp1['qH2O'][:])

# %%
