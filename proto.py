# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import math
import matplotlib.pyplot as plt
import numpy as np
import tools as to
import matplotlib
from matplotlib import rc
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.fftpack.realtransforms import dct
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.mixture import GaussianMixture
#data=np.load('lab1_data.npz')['data']
#example = np.load('lab1_example.npz')['example'].item()
#r=data[1]
#a=r['samples']
#font = {'family' : 'normal',
#        'size'   : 16}

#matplotlib.rc('font', **font)
def collectmfcc(data):
    length=data.size

    for i in range(length):
        if i==0:
            tot=mfcc(data[i]['samples'])
            n,m=mfcc(data[i]['samples']).shape
            l=[n]
            mdict={i:tot}
        tot=np.vstack((tot,mfcc(data[i]['samples'])))
        n,m=mfcc(data[i]['samples']).shape
        l.append(n)
        mdict[i]=tot
    return tot,l,mdict

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    #plt.rc('text',usetex=True)
    #plt.pcolormesh(frames)
    #plt.axis('off')
    #plt.title('Enframed Samples',fontsize=20)
    #plt.show()

    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    #plt.pcolormesh(ceps)
    #plt.title('Mfcc Coefficients',fontsize=20)
    #plt.axis('off')
    #plt.show()
    lmfcc=to.lifter(ceps,liftercoeff)
    #corr=np.corrcoef(lmfcc)
    #plt.pcolormesh(corr)
    #plt.show()
    return lmfcc,mspec

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    sampsize=samples.size
    N=math.floor(samples.size/(winlen-winshift))
    samp_enframe=np.zeros([N,winlen])
    i=0
    while(samples[i*(winlen-winshift):i*(winlen-winshift)+winlen].shape[0]==winlen):
        samp_enframe[i,:]=samples[i*(winlen-winshift):i*(winlen-winshift)+winlen]
        i=i+1
    #plt.pcolormesh(samp_enframe)
    #plt.show()
    return samp_enframe[0:-1,:]

def preemp(inp, p=0.97):
    A=1
    B=[1,-p]
    N,M=inp.shape
    presignal=np.zeros([N,M])
    for i in range(N):
        presignal[i,:]=signal.lfilter(B,A,inp[i,:])
    return presignal
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """







def windowing(inp):
    window=signal.hamming(400,sym=False)
    N,M=inp.shape
    presignal=np.zeros([N,M])
    for i in range(N):
        presignal[i,:]=window*inp[i,:]
    return presignal

    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

def powerSpectrum(inp, nfft):
    power=abs(fft(inp,nfft))**2
    return power
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

def logMelSpectrum(inp, samplingrate=20000):
    filters=to.trfbank(samplingrate,512)
    #print(inp.shape)
    #print(filters)
    #print(filters.shape)
    #plt.plot(np.transpose(filters))
    #plt.title('Mel Filterbank',fontsize=20)
    #plt.plot(filters[1,:])
    #plt.show()
    #plt.show()
    mel=np.log(np.matmul(inp,np.transpose(filters)))
    #print(mel)
    return np.log(np.matmul(inp,np.transpose(filters)))
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
#a=logMelSpectrum(p)
def cepstrum(inp, nceps=13):
    return(dct(inp,type=2,axis=1,norm='ortho')[:,0:nceps])

    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

def euclid(cep1,cep2):
    cep1.shape
    i1,l=cep1.shape
    i2,l2=cep2.shape
    #print(i1,i2)
    dist=np.zeros([i1,i2])
    #print(dist.shape)

    for i in range(i1):
        for j in range(i2):
            dist[i,j]=math.sqrt(np.matmul(np.transpose(cep1[i,:]-cep2[j,:]),cep1[i,:]-cep2[j,:]))
            #print(dist)
    return dist
    #plt.pcolormesh(dist)
    #plt.show()



def dtw(x, y, dist=0):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    distances=euclid(x,y)
    d,path=fastdtw(x,y,dist=euclidean)
    norm_scale=x.size+y.size
    d=d/norm_scale
    return d

def alldtw(data,ndata=44):
    distmatrix=np.zeros([ndata,ndata])
    for i in range(ndata):
        print(i)
        for j in range(ndata):
            distmatrix[i,j]=dtw(mfcc(data[i]['samples']),mfcc(data[j]['samples']))
    return distmatrix

def gauss(data):
    obj=GaussianMixture(n_components=8,covariance_type='diag')
    obj.fit(data)
    return obj

def corr(allmfcc):
    co=np.corrcoef(allmfcc)
    plt.pcolormesh(co)
    plt.title('Correlation Coefficients',fontsize=20)
    plt.axis('off')
    plt.show()





#al,l,mdict=collectmfcc(data)



#corr(al)
#obj=gauss(al)
#r=obj.predict_proba(mdict[2])
#e=obj.predict_proba(mdict[3])
#plt.pcolormesh(r)
#plt.show()
#plt.pcolormesh(e)
#plt.show()



#a=mfcc(example['samples'])
#a1=mfcc(data[42]['samples'])
#a2=mfcc(data[43]['samples'])
#d=dtw(a1,a2)
#dmat=alldtw(data)
#r=to.tidigit2labels(data)
#plt.pcolormesh(dmat)
#plt.title('Euclidian Distances',fontsize=20)
#plt.axis('off')
#plt.show()

#Z=linkage(dmat)
#dn=dendrogram(Z)
#plt.show()
#print(d)
#a=collectmfcc(data)
#e=example
