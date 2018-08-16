import numpy as np
from lab3_tools import *
from proto import mfcc
import os
from prondict import prondict
from proto2 import concatHMMs, viterbi
from sklearn.mixture import log_multivariate_normal_density
import matplotlib.pyplot as plt
from tools2 import log_multivariate_normal_density_diag
import random
from sklearn.preprocessing import StandardScaler


np.set_printoptions(threshold=np.nan)

def words2phones(wordList, pronDict, addSilence=True, addShortPause=False):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phonelist=['sil']
    for word in wordList:
        for element in prondict[word]:
            phonelist.append(element)
    phonelist.append('sil')
    return phonelist


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    utteranceHMM=concatHMMs(phoneHMMs,phoneTrans)
    emission=log_multivariate_normal_density_diag(lmfcc,utteranceHMM['means'],utteranceHMM['covars'])
    vitpath,vitmax=viterbi(emission,np.log(utteranceHMM['startprob']),np.log(utteranceHMM['transmat']))


    return vitpath,vitmax





def getfeatures():
    """
    Fetches MFCC auditory feature vectors from the TIDIGITS dataset and alignes them with their corresponding 
    phonetic transcription. The data is saved as an npz-file.
    """
    traindata = []
    for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                print(filename)
                samples, samplingrate = loadAudio(filename)
                lmfcc,mspec = mfcc(samples)
                wordTrans = list(path2info(filename)[2])
                phoneTrans1=words2phones(wordTrans,prondict)
                phoneTrans=phoneTrans1[1:-1]
                utteranceHMM=concatHMMs(phoneHMMs,phoneTrans1)
                stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans1
                                  for stateid in range(nstates[phone])]

                vit,v=forcedAlignment(lmfcc,phoneHMMs,phoneTrans)
                viterbiStateTrans=[]
                for i in vit[1]:
                    viterbiStateTrans.append(stateTrans[i])
                targets=[]
                for r in viterbiStateTrans:
                    targets.append(stateList.index(r))

                traindata.append({'filename': filename, 'lmfcc': lmfcc,
                                  'mspec': mspec, 'targets': targets})
    np.savez('traindata.npz', traindata=traindata)


def sortmanwoman(alldata):
    """
    Splits female and male speaker features into two dictionaries.
    (For MFCC features)
    """
    dman={}
    dwoman={}
    nr_data=alldata.shape[0]
    for i in range(nr_data):
        filename=alldata[i]['filename']
        if 'woman' in filename:
            speaker=filename[41:43]+'_w'
            if speaker in dwoman:
                dwoman[speaker].append(alldata[i])
            else:
                dwoman[speaker]=[alldata[i]]
        else:
            speaker=filename[39:41]+'_m'
            if speaker in dman:
                dman[speaker].append(alldata[i])
            else:
                dman[speaker]=[alldata[i]]
    return dman,dwoman



def sortmanwoman2(alldata):
    """
    Splits female and male speaker features into two dictionaries.
    (For mel filterbank features.)
    """
    dman={}
    dwoman={}
    nr_data=alldata.shape[0]
    for i in range(nr_data):
        filename=alldata[i]['filename']
        if 'woman' in filename:
            speaker=filename[40:42]+'_w'
            if speaker in dwoman:
                dwoman[speaker].append(alldata[i])
            else:
                dwoman[speaker]=[alldata[i]]
        else:
            speaker=filename[38:40]+'_m'
            if speaker in dman:
                dman[speaker].append(alldata[i])
            else:
                dman[speaker]=[alldata[i]]
    return dman,dwoman




def reg(alldata):
    """
    Normalizes the features.
    """
    nr_data=alldata.shape[0]
    normdata=alldata
    for i in range(nr_data):
        utterance=alldata[i]
        scaler=StandardScaler()
        scaler.fit(utterance['lmfcc'])
        normdata[i]['lmfcc']=scaler.transform(utterance['lmfcc'])
        scaler2=StandardScaler()
        scaler2.fit(utterance['mspec'])
        normdata[i]['mspec']=scaler2.transform(utterance['mspec'])
    return normdata


def gettraintest(data):
    lmfcc_x=np.empty([0,13])
    mspec_x=np.empty([0,40])

    targets_y=np.empty([0,1])
    for keys in data:
        for utterance in data[keys]:
            lmfcc=utterance['lmfcc']
            mspec=utterance['mspec']
            target=np.array(utterance['targets'])
            target=np.reshape(target,[target.shape[0],1])
            lmfcc_x=np.vstack((lmfcc_x,lmfcc))
            mspec_x=np.vstack((mspec_x,mspec))
            targets_y=np.vstack((targets_y,target))
    return lmfcc_x,mspec_x,targets_y





def createtestset(alldata, flag = False):
    """
    Splits data into test and train data.
    """
    traindict={}
    testdict={}
    l1=[]
    l2=[]

    if flag:
        d1,d2=sortmanwoman(alldata)
    else:
        d1,d2=sortmanwoman2(alldata)

    for keys in d1:
        l2.append(keys)
    for keys in d2:
        l1.append(keys)

    for key in l2[0:len(d1)//10 +1]:
        testdict[key] = d1[key]
    for key in l1[0:len(d1)//10 +1]:
        testdict[key] = d2[key]

    for key in l2[len(d1)//10 +1:]:
        traindict[key] = d1[key]
    for key in l1[len(d1)//10 +1:]:
        traindict[key] = d2[key]
    return traindict,testdict


