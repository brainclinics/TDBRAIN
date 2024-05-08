#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:06:04 2019

@author: hanneke
"""
import os
import pickle
import numpy as np

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def loadbysubID(root_dir, subID, condition):
    """*---------------------------------------------------------------------*
    This function loads the preprocessed data of a single subject by IDcode
    and condition

    use as data = loadbysubID(sourcepath, subID, condition)
        * root_dir: string of the path the data is at (this could be any level
                                                         above specific datasets,
                                                         faster when closer to
                                                         actual file location)
        * subID: 8 digit ID code of the subject
        * condition: condition you want to load e.g 'EO' or 'EC'
    returns:
        an object containing EEG data, EEG labels, information aabout previous
        processing and functions
    *-----------------------------------------------------------------------*"""
    import fnmatch
    from BCD_gitprojects.data_processing.processing.interprocessing import interdataset as ids
    def find(IDcode, pattern, path): #This function walks throug all folders to find the specific IDcode
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if IDcode in name and fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    filepaths = find(subID,'*.npy', root_dir)

    inname = [f for f in filepaths if condition in f and not 'BAD' in f][0]

    print(inname)
    with open(inname,'rb') as input: preproc = pickle.load(input)

    return ids(preproc)

class FilepathFinder(object):
    """*---------------------------------------------------------------------*
    This object class collects all filepaths of data of interest defined by a pattern
    (such as an extension) that are within the root_dir, excluding folders or datatypes
    if necessary, for use in DataLoader which will in its turn be used in
    tensorflow.keras.Model.fit_generator

    use as filepaths = FilepathFinder(pattern, root_dir, exclude, test_size=0.1)
        * root_dir: string of the path the data is at (this could be any level
                                                         above specific datasets,
                                                         faster when closer to
                                                         actual filelocations)
        * pattern: specifics about the files you want to include
        * exclude: an array of specific folders or file patterns you don't want
        to include
        * test_size is optional if you want to do an sklearn.model_selection.GroupShuffleSplit
        sklearn.model_selection.GroupShuffleSplit is built into this object and takes
        in the IDcodes as groups and test_size as test_size

    !! Note that right now, this object explicitly only takes in first sessions
    *-----------------------------------------------------------------------*"""
    import os, pickle
    def __init__ (self, pattern, root_dir):
        self.pattern = pattern
        self.root_dir = root_dir

    def get_filenames(self, sessions='all'):
        """returns and array of all filepaths adhering to the selected pattern
        and root_dir  in <data>.selectedfiles"""
        self.files = self.__find()

    def __find(self):
        """subfunction walking through all levels of subfolders and finds files consistent
        with the pattern given """
        import  os
        result = []
        for root, dirs, files in os.walk(self.root_dir):
            for name in files:
                if self.pattern in name:
                    #print('yes!')
                    #print(self.pattern)
                    result.append(os.path.join(root, name))
        print(str(len(result)) + ' files listed')
        return result

#%% fileloader template
"""
exclude = ['Apple','DS','._']; s=[]
subs = [s for s in os.listdir(sourcepath) if os.path.isdir(os.path.join(sourcepath,s)) and not any([excl in s for excl in exclude])]
subs = np.sort(subs)

if varargs['condition']:
    reqconds = varargs['condition']
else:
    reqconds = ['EO','EC']

k=startsubj
if subject == None:
    subarray = np.arange(k,len(subs))
elif type(subject) ==int:
    subarray = [subject]
elif type(subject) == str:
    subarray = np.array([np.where(subs==subject)[0]][0])

files=[]
for s in subarray:
    sessions = [session for session in os.listdir(os.path.join(sourcepath,subs[s])) if not any([excl in session for excl in exclude]) and os.path.isdir(os.path.join(sourcepath,subs[s],session))]
    for sess in range(len(sessions)):
        conditions = []
#        np.sort([s for s in IDs if not any(xs in s for xs in subs)])
        allconds = np.array([conds for conds in os.listdir(os.path.join(sourcepath,subs[s],sessions[sess])) if ('.csv' in conds) and not any([excl in conds for excl in exclude])])
        if reqconds == 'all':
            conditions = allconds
        else:
            conditions = np.array([conds for conds in allconds if any([a.upper() in conds for a in reqconds])])
            for c in range(len(conditions)):
"""

