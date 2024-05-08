#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:55:28 2020
@author: hannekevandijk
"""

"""{
Import necesarry modules
}"""

import os
import sys
import pickle as pickle
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
#from IPython import embed; embed()

from BRAIN_code.autopreprocess_pipeline import autopreprocess_standard
from BRAIN_code.inout import FilepathFinder
from BRAIN_code.interprocessing import interdataset as intds


def end2end_alphaPowerandiAPF(varargs):
    """
    This function performs end-to-end processing of the neurophysiological validation
    presented in the TD-BRAIN database manuscript

    Parameters
    ----------
    varargs : dictionary
            {'sourcepath': <path where the raw data are>,
             'participantspath':<path were the participants.tsv file is saved>''
            'preprocpath': <path where you want to save the preprocessed data>,
            'resultspath: <path where you want to save the results>'}
            'condition': ['EO','EC']
            'chans': <EEG channel to be post-processed for frequency analysis, default ='Pz'>

    Returns
    -------
    TYPE
        plots with results, statistics (printed out)
    TYPE
        preprocessed data saved in standardized database, results saved in dictionary
    """

    """{
    1rst: Preprocessing using defaults
    }"""

    autopreprocess_standard(varargs)

    """{
    2nd: segmentation and frequency analysis
    }"""

    if not 'preprocpath' in varargs:
        raise ValueError('preprocpath not defined')

    preprocs = FilepathFinder('.npy', varargs['preprocpath'])
    preprocs.get_filenames()

    if not 'participantspath' in varargs:
        raise ValueError('participantspath not defined')

    tsvdat = pd.read_csv(varargs['participantspath']+'/participants.tsv',sep = '\t')

    """ compute powerspectrum and make dictionary with results that will be saved in varrags['resultspath']"""
    def computeFFT(data, Fs=500):
        from scipy.signal import hann
        hannwin = np.array(hann(data.shape[-1]))
        power = np.squeeze(np.mean(np.abs(np.fft.fft(data*hannwin))**2,axis=0))
        freqs = np.linspace(0,Fs/2,np.int(len(power)/2)) #power1
        return power[:len(freqs)], freqs

    output = {'IDcodes':['a'],
              'conds':[99],
              'session': [99],
              'age': [199],
              'sex': [99]}
    s = 0;missingsession=[]
    for f in preprocs.files:
        #print(s, f.rsplit('_')[5])
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
        output['conds'].append(f.rsplit('_')[-4])
        output['power'] = np.vstack((output['power'],np.squeeze(computeFFT(chandata)[0])))
        try:
            idx = np.where((tsvdat['subID']==np.int(sid)) & (tsvdat['sessID']==np.int(sessid)))[0][0]
            output['age'].append(tsvdat['age'][idx])
            output['sex'].append(tsvdat['gender'][idx])
        except:
            print(f)
            missingsession.append(f)
            output['age'].append(np.nan)
            output['sex'].append(np.nan)
        s = s+1
    output['freqs'] = computeFFT(chandata)[1]
    df = pd.DataFrame(missingsession)
    df.to_csv(varargs['resultspath']+'missingsessionsinpo.csv', sep=',')
    with open (varargs['resultspath'] + '/outputAlphaPower.npy', 'wb') as saveoutput:  # Overwrites any existing file.
        pickle.dump(output, saveoutput, -1)
    saveoutput.close()


    age = np.squeeze(np.array(output['age'][1:]));gender = np.array(output['sex'][1:]);
    power = np.array(output['power'][1:]); conds = np.array(output['conds'][1:]);
    idcodes = np.array(output['IDcodes'][1:]); freqs= output['freqs']

    foi = [(np.where((freqs>=6.9) & (freqs<=13.1)))][0][0]

    """ compute mean log power for Eyes open and Eyes closed and perform statistics """
    #select data for EC and EO from the entire database
    p=0;meanEC=[];meanEO=[]
    for s in np.unique(idcodes):
        idxs=[0]
        try:
            idxs = [(np.where((idcodes==s) & (conds=='EC')))[0][0],(np.where((idcodes==s) & (conds=='EO')))[0][0]]
        except:
            pass
        #print(len(idxs))
        if len(idxs)==2:
            meanEC.append(np.mean(np.log(power[idxs[0],foi])))
            meanEO.append(np.mean(np.log(power[idxs[1],foi])))
    # dependent t-test
    from scipy import stats
    t,p = stats.ttest_rel(meanEC,meanEO)
    print('dependent t-test EC vs EO: t: '+str(np.round(t,decimals=2))+' p: ',str(np.round(p,decimals=2)))
    # cohens d
    diff = np.mean(meanEC)-np.mean(meanEO)
    pooledstdv = np.sqrt((np.std(meanEC)**2+np.std(meanEO)**2)/2)
    cohend = diff/pooledstdv
    print('Effect size (cohens d): '+str(np.round(cohend,decimals=2)))
    """ plotting FFT EyesClosed vs EyesOpen """
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
    plt.show()

    """ plotting iAPF and model plus statistucs """
    #function for loggaussian model
    #only for Eyes closed
    condidx = np.where(conds =='EC')[0]
    age1 = age[condidx]
    powa = power[condidx]

    if len(np.where(np.isnan(age1))[0])>0:
        X = age1[~np.isnan(age1)];
        pow1 = powa[~np.isnan(age1)]
    else:
        X = age1
        pow1 = powa

    apf = [];# np.zeros((len(X)));apf[:]=np.nan
    ageidx=[]
    pw=[];pwage=[];apfage=[]
    ageidx = np.argsort(X)
    powersorted = np.log(pow1[ageidx,:])
    for lva in range(powersorted.shape[0]):
        if np.mean(powersorted[lva,foi])>13.5:#determined by Helena as well, LVA means no alpha or little present
            pw.append(powersorted[lva,:])
            pwage.append(X[lva])
    pw = np.array(pw)
    for p in range(pw.shape[0]):
        peaks_raw = find_peaks(pw[p,foi], height = np.max(pw[p,foi])*0.4,threshold=np.ones(foi.shape)*0.05)
        if len(peaks_raw[0])>0:
            maxidx = peaks_raw[0][np.argmax(peaks_raw[1]['peak_heights'])] # find the peak with the maximum amplitude
            apf.append(np.round(freqs[foi[maxidx]],decimals=1))
            apfage.append(pwage[p])

    data = np.array([np.sort(apfage),apf])
    set1 = pd.DataFrame(data.T,columns=['age','iapf'])

    sns.lmplot(x='age',y='iapf',data = set1,fit_reg=False,hue='data', palette=colors)
    plt.title('iAPF')
    plt.show()

    """ plot age distributions for males and females """

    maleages = np.unique(tsvdat['age'][tsvdat['gender']==1])
    femaleages = np.unique(tsvdat['age'][tsvdat['gender']==0])

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
            ax.set_yticklabels(list([0,10,20,30,40,50,60,70,80,90]),rotation=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        g=g+1

    plt.suptitle('age distribution')
    plt.subplots_adjust(wspace=0.22)
    plt.show()

if __name__ == "__main__":
    varargs = {}
    varargs['sourcepath'] = sys.argv[1]+ 'derivatives/'
    print('Reading data from: '+ str(sys.argv[1]) + 'derivatives/')
    varargs['participantspath'] = sys.argv[1]
    print('Reading data from: '+ str(sys.argv[1]))

    varargs['preprocpath'] = varargs['sourcepath']+'preprocessed'
    if not os.path.exists(varargs['preprocpath']):
        os.mkdir(varargs['preprocpath'])
    print('Writing preprocessed data to: '+varargs['preprocpath'])
    varargs['resultspath'] = varargs['sourcepath']+'results_manuscript'
    if not os.path.exists(varargs['resultspath']):
        os.mkdir(varargs['resultspath'])
    print('Writing results to: '+varargs['resultspath'])

    varargs['condition']=['EO','EC']
    varargs['chans']='Pz'

    end2end_alphaPowerandiAPF(varargs)














