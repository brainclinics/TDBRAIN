#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:55:28 2020

@author: hannekevandijk
"""
#%% 1
from BCD_gitprojects.preprocessing_beta.io import FilepathFinder
from BCD_gitprojects.preprocessing_beta.interprocessing import interdataset as intds
import numpy as np
import pickle5 as pickle
import os
import pandas as pd
import numpy as np

root_dir = '/NAS/database/BCD_OA_database/'
exclude = ['BAD', '._' ,'Apple']
files = FilepathFinder('.npy', root_dir+'BCD_OA_preprocessed_recoded_trial_set/', exclude =exclude)
files.__get_filenames__()

tsvdat = pd.read_csv(root_dir+'BCD_OA_edf_recoded_trial_set/participants.tsv',sep = '\t')

def computeFFT(data):
    from scipy.signal import hann
    hannwin = np.array(hann(data.shape[-1]))
    power = np.squeeze(np.mean(np.abs(np.fft.fft(data*hannwin))**2,axis=0))
    freqs = np.linspace(0,varargs['fs']/2,np.int(len(power)/2)) #power1
    return power[:len(freqs)], freqs




varargs = {'chans': 'Pz', 'fs':500}


#%% compute iAPF and powerspectrum
output = {'IDcodes':['a'],
          'conds':[99],
          'session': [99],
          'age': [199],
          'sex': [99]}
s = 0
for f in files.selectedfiles:
    print(s)
    with open(f,'rb') as input: preproc = pickle.load(input)
    tmpdat = intds(preproc)
    ch = np.where(tmpdat.labels == varargs['chans'])[0][0]
    tmpdat.segment(trllength=5)
    chandata = np.array(tmpdat.data[:,ch,:])
    if s==0:
        output['power']=np.zeros((len(computeFFT(chandata)[0])))
    sid = f.rsplit('/')[-2][:-2]; sessid = f.rsplit('/')[-2][-1:]
    output['IDcodes'].append(sid)
    output['session'].append(sessid)
    output['conds'].append(f.rsplit('_')[-2])
    #read belonging tsvdat data
    idx = np.where((tsvdat['subID']==np.int(sid)) & (tsvdat['sessID']==np.int(sessid)))[0][0]
    output['age'].append(tsvdat['age'][idx])
    output['sex'].append(tsvdat['gender'][idx])
    output['power'] = np.vstack((output['power'],np.squeeze(computeFFT(chandata)[0])))
    s = s+1
output['freqs'] = computeFFT(chandata)[1]


with open(root_dir + '/outputAlphaPower.npy', 'wb') as saveoutput:  # Overwrites any existing file.
    pickle.dump(output, saveoutput, -1)
saveoutput.close()

#%% Power Eyes Closed vs Eyes Open
import numpy, scipy, pandas
import pickle5 as pickle

infile = open(root_dir + 'outputAlphaPower.npy', 'rb')
output = pickle.load(infile)
# #%% plotting iAPF
import seaborn as sns
age = np.squeeze(np.array(output['age'][1:]));gender = np.array(output['sex'][1:]); power = np.array(output['power'][1:]); conds = np.array(output['conds'][1:]); idcodes = np.array(output['IDcodes'][1:])

freqs= output['freqs']
foi = [(np.where((freqs>=6.9) & (freqs<=13.1)))][0][0]

p=0;meanEC=[];meanEO=[]
for s in np.unique(idcodes):
    #print(s)
    idxs=[0]
    try:
        idxs = [(np.where((idcodes==s) & (conds=='EC')))[0][0],(np.where((idcodes==s) & (conds=='EO')))[0][0]]
    except:
        pass
    print(len(idxs))
    if len(idxs)==2:
        meanEC.append(np.mean(np.log(power[idxs[0],foi])))
        meanEO.append(np.mean(np.log(power[idxs[1],foi])))
# dependent t-test
from scipy import stats
t,p = stats.ttest_rel(meanEC,meanEO)
# cohens d
diff = np.mean(meanEC)-np.mean(meanEO)
pooledstdv = np.sqrt((np.std(meanEC)**2+np.std(meanEO)**2)/2)
cohend = diff/pooledstdv

#%% plotting FFT EyesClosed vs EyesOpen
foirange = [(np.where((freqs>=2) & (freqs<=45)))][0][0]

EC = np.where(conds=='EC')[0]
EO = np.where(conds=='EO')[0]
ECpower = power[EC,:]
EOpower = power[EO,:]
data1 = np.vstack((freqs[foirange],np.mean(np.log(EOpower[:,foirange]),axis=0)));set1 = pd.DataFrame(data1.T,columns=['frequency (Hz)','logPower'])
data2 = np.array([freqs[foirange],np.mean(np.log(ECpower[:,foirange]),axis=0)]);set2 = pd.DataFrame(data2.T,columns=['frequency (Hz)','logPower'])
concatenated = pd.concat([set1.assign(condition='EO'), set2.assign(condition='EC')])

colors = list(['lightseagreen','darkblue'])
sns.lineplot(x='frequency (Hz)',y='logPower',data=concatenated, hue= 'condition',palette = colors, linewidth = 2, alpha=0.8)


#%% iAPF and model
def logGauss(A, G, lnSD, S, X):
# S = offset, A=A, X=age[s], G= GeoMean, lnSD = lnGeoSD; math.log is natural log in python
    return S+((A/X)*np.exp(-0.5*(np.log(X/G)/lnSD)**2))

#These are curve parameters determined by Helena (PhD student) based on our large >4000 EEG Eyes closed dataset
A = 212
G = 163
lnSD = 1.35
S=6.5

idx = np.where(conds =='EC')[0]

X = age
pow1 = power

from scipy.signal import find_peaks
import matplotlib.pyplot as plt

apf = np.zeros((len(X)));apf[:]=np.nan
ageidx=[]
pw=[]
ageidx = np.argsort(X)
powersorted = np.log(pow1[ageidx,:])
for lva in range(powersorted.shape[0]):
    if np.mean(powersorted[lva,foi])>13.5:#determined by Helena as well, LVA means no alpha or little present
        pw.append(powersorted[lva,:])
pw = np.array(pw)
for p in range(pw.shape[0]):
    peaks_raw = find_peaks(pw[p,foi], height = np.max(pw[p,foi])*0.4,threshold=np.ones(foi.shape)*0.05)
    if len(peaks_raw[0])>0:
        maxidx = peaks_raw[0][np.argmax(peaks_raw[1]['peak_heights'])] # find the peak with the maximum amplitude
        apf[p] =np.round(freqs[foi[maxidx]],decimals=1)

data = np.array([np.sort(X),apf])
set1 = pd.DataFrame(data.T,columns=['age','iapf'])

model = logGauss(A,G,lnSD,S,np.sort(X))
datamodel = np.array([np.sort(X),model])
set2 = pd.DataFrame(datamodel.T,columns=['age','iapf'])
concatenated = pd.concat([set1.assign(data='iapf'), set2.assign(data='model')])
sns.lmplot(x='age',y='iapf',data = concatenated,fit_reg=False,hue='data', palette=colors)
ss_res = np.sum((apf - model) ** 2)
ss_tot = np.sum((apf - np.mean(apf)) ** 2)
r2 = np.round(1 - (ss_res / ss_tot),decimals=2)
plt.text(70,12.5,'r2: '+str(r2))
plt.title('iAPF and model')


#%% age distribution of both genders

maleages = tsvdat['age'][tsvdat['gender']==1]
femaleages = tsvdat['age'][tsvdat['gender']==0]

ages = {0:femaleages,
        1:maleages}

fig, axis = plt.subplots(nrows=1, ncols = 2, sharey=False,figsize=(5,5))
g=0
colors = np.array(['lightseagreen','darkblue'])
for ax in axis.ravel():
    ax.hist(np.array(ages[g]),bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90],color=colors[g], orientation='horizontal', alpha=0.7, edgecolor='white', linewidth=0.2)
    if g==0:
        ax.set_xlabel('Female count')
        ax.invert_xaxis()
        ax.set_yticks(np.arange(0,100,10))
        ax.set_yticklabels([])
        ax.yaxis.set_ticks_position('right')

        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        ax.set_xlabel('Male count')
        ax.set_yticks(np.arange(0,100,10))
        ax.set_yticklabels(list(['',10,20,30,40,50,60,70,80,90,100]),rotation=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    g=g+1

plt.suptitle('age distribution')
plt.subplots_adjust(wspace=0.22)














