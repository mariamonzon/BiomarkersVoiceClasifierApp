### Utils imported from
# WGANSing: A Multi-Voice Singing Voice Synthesizer Based on the Wasserstein-GAN
# https://github.com/MTG/WGANSing


import sys
import os,re
import collections
import csv
import soundfile as sf
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
from vocoder import extract_sp_world, extract_ap_world, gen_wave_world
# from acoufe import pitch
import librosa
from tqdm import tqdm

import pyworld
import numpy as np
import scipy.interpolate
import pysptk


def extract_sp_world(x, f0, sr, hoptime, fft_size=None):
    # NOTE:
    # F0 values too low for used FFT size are handled as unvoiced in CheapTrick.
    # If this happens, we warn the user.

    f0 = f0.squeeze()
    f0 = np.ascontiguousarray(f0, dtype=np.float64)  # pyworld requires C-contiguous float64 array

    # warn for very low f0
    # fft_size = pyworld.get_cheaptrick_fft_size(sr) if fft_size is None else fft_size
    # f0_floor = pyworld.get_cheaptrick_f0_floor(sr, fft_size)
    # n_f0_too_low = int(np.sum(np.logical_and(f0 > 0, f0 <= f0_floor)))
    # if n_f0_too_low > 0:
    #     warnings.warn('F0 too low (<= {:.2f}) for FFT size ({:d}) for {:d} samples'.format(f0_floor, fft_size, n_f0_too_low))

    n_frames = f0.shape[0]
    t = np.arange(n_frames) * hoptime

    sp = pyworld.cheaptrick(x, f0, t, sr)

    if not np.all(np.isfinite(sp)):
        raise ValueError('Configuration or input signal caused NaNs in WORLD CheapTrick analysis')
        # NOTE: This seems to happen occassionally, e.g. taking 16kHz TIMIT audio, upsampling
        # it to 32kHz and performing WORLD analysis

    sp = 10 * np.log10(sp)  # power to decibels

    return sp


def extract_ap_world(x, f0, sr, hoptime, fft_size=None, fill_unvoiced=True):
    # NOTE:
    # F0 is used in D4C algorithm, but simply clipped to internal `lowest_f0` (hardcoded to 40.0 Hz);
    # `lowest_f0` is also used to determine D4C's internal FFT size. For unvoiced frames, D4C just
    # results in ap[:] = 1. So no need to handle low `f0` values specially here.

    f0 = f0.squeeze()
    f0 = np.ascontiguousarray(f0, dtype=np.float64)  # pyworld requires C-contiguous float64 array

    n_frames = f0.shape[0]
    t = np.arange(n_frames) * hoptime

    ap = pyworld.d4c(x, f0, t, sr)

    if not np.all(np.isfinite(ap)):
        raise ValueError('Configuration or input signal caused NaNs in WORLD D4C analysis')

    ap = 10. * np.log10(ap ** 2)  # linear to decibels

    if fill_unvoiced:
        # d4c sets aperodicity to 1 for unvoiced regions,
        # to avoid discontinuities and allow changing f0 we
        # fill these regions by extrapolating
        is_voiced = f0 > 0.0
        if not np.any(is_voiced):
            pass  # all unvoiced, do nothing
        else:
            for k in range(ap.shape[1]):
                ap[~is_voiced, k] = np.interp(np.where(~is_voiced)[0], np.where(is_voiced)[0], ap[is_voiced, k])
        # interp_f = scipy.interpolate.interp1d(np.where(is_voiced)[0], ap[is_voiced, :], kind='linear', axis=0, assume_sorted=True, bounds_error=False, fill_value='extrapolate')
        # ap[~is_voiced, :] = interp_f(np.where(~is_voiced)[0])

    return ap


def gen_wave_world(f0, sp, ap, sr, hoptime):
    # NOTE:
    # No need to handle very low or high `f0` values specially for
    # synthesis.

    f0 = f0.squeeze()
    sp = 10 ** (sp / 10)
    ap = 10 ** (ap / 20)
    ap[f0 == 0, :] = 1.0  # probably has no effect
    ap = np.clip(ap, 0.0, 1.0)  # WORLD fails catastrophically for out of range aperiodicity
    f0 = np.ascontiguousarray(f0, dtype=np.float64)
    sp = np.ascontiguousarray(sp, dtype=np.float64)
    ap = np.ascontiguousarray(ap, dtype=np.float64)
    x = pyworld.synthesize(f0, sp, ap, sr, hoptime * 1000.0)

    return x

def sp_to_mgc(sp, ndim, fw, noise_floor_db=-120.0):
    # HTS uses -80, but we shift WORLD/STRAIGHT by -20 dB (so would be -100); use a little more headroom (SPTK uses doubles internally, so eps 1e-12 should still be OK)
    dtype = sp.dtype
    sp = sp.astype(np.float64)  # required for pysptk
    mgc = np.apply_along_axis(pysptk.mcep, 1, np.atleast_2d(sp), order=ndim - 1, alpha=fw, maxiter=0, etype=1,
                              eps=10 ** (noise_floor_db / 10), min_det=0.0, itype=1)
    if sp.ndim == 1:
        mgc = mgc.flatten()
    mgc = mgc.astype(dtype)
    return mgc


def mgc_to_sp(mgc, spec_size, fw):
    dtype = mgc.dtype
    mgc = mgc.astype(np.float64)  # required for pysptk
    fftlen = 2 * (spec_size - 1)
    sp = np.apply_along_axis(pysptk.mgc2sp, 1, np.atleast_2d(mgc), alpha=fw, gamma=0.0, fftlen=fftlen)
    sp = 20 * np.real(sp) / np.log(10)
    if mgc.ndim == 1:
        sp = sp.flatten()
    sp = sp.astype(dtype)
    return sp


def mgc_to_mfsc(mgc):
    is_1d = mgc.ndim == 1
    mgc = np.atleast_2d(mgc)
    ndim = mgc.shape[1]

    # mirror cepstrum
    mgc1 = np.concatenate([mgc[:, :], mgc[:, -2:0:-1]], axis=-1)

    # re-scale 'dc' and 'nyquist' cepstral bins (see mcep())
    mgc1[:, 0] *= 2
    mgc1[:, ndim - 1] *= 2

    # fft, truncate, to decibels
    mfsc = np.real(np.fft.fft(mgc1))
    mfsc = mfsc[:, :ndim]
    mfsc = 10 * mfsc / np.log(10)

    if is_1d:
        mfsc = mfsc.flatten()

    return mfsc


def sp_to_mfsc(sp, ndim, fw, noise_floor_db=-120.0):
    # helper function, sp->mgc->mfsc in a single step
    mgc = sp_to_mgc(sp, ndim, fw, noise_floor_db)
    mfsc = mgc_to_mfsc(mgc)
    return mfsc


def get_warped_freqs(ndim, sr, fw):
    # warped frequencies
    f = np.linspace(0.0, sr / 2, ndim)
    w = 2 * np.pi * f / sr
    a = -fw  # XXX: why negative?
    w_warped = np.arctan2((1 - a ** 2) * np.sin(w), (1 + a ** 2) * np.cos(w) - 2 * a)
    f_warped = w_warped / (2 * np.pi) * sr
    return f_warped


def mfsc_to_mgc(mfsc):
    # mfsc -> mgc -> sp is a much slower alternative to mfsc_to_sp()
    is_1d = mfsc.ndim == 1
    mfsc = np.atleast_2d(mfsc)
    ndim = mfsc.shape[1]

    mfsc = mfsc / 10 * np.log(10)
    mfsc1 = np.concatenate([mfsc[:, :], mfsc[:, -2:0:-1]], axis=-1)
    mgc = np.real(np.fft.ifft(mfsc1))
    mgc[:, 0] /= 2
    mgc[:, ndim - 1] /= 2
    mgc = mgc[:, :ndim]

    if is_1d:
        mgc = mgc.flatten()

    return mgc


def mfsc_to_sp(mfsc, f_warped, spec_size, sr):
    # reconstruct sp by interpolation
    # alternative is mfsc_to_mgc(), mgc_to_sp() which uses SPTK's mgc2sp
    # interpolation is quite accurate if high mfsc dimensionality is used (e.g. 60)

    is_1d = mfsc.ndim == 1
    mfsc = np.atleast_2d(mfsc)
    ndim = mfsc.shape[1]

    f_sp = np.linspace(0, sr / 2, spec_size)
    interp_f = scipy.interpolate.CubicSpline(f_warped, mfsc, axis=-1, bc_type='clamped', extrapolate=None)
    sp = interp_f(f_sp)

    if is_1d:
        sp = sp.flatten()

    return sp


# world band aperiodicities
wbap_interval = 3000.0
wbap_fmax = 15000.0
wbap_eps = 1e-12


def get_num_wbap(sr):
    return int(np.floor(min(wbap_fmax, sr / 2 - wbap_interval) / wbap_interval))


def get_wbap_freqs(n_wbap, with_edges=False, sr=None):
    if with_edges:
        return np.hstack([0, np.arange(1, 1 + n_wbap) * wbap_interval, sr / 2])
    else:
        return np.arange(1, 1 + n_wbap) * wbap_interval


def ap_to_wbap(ap, n_wbap, sr):
    # ap (in dB) is linear bpf at regular intervals, e.g. 3k, 6k, 9k, 12k,
    # with points at DC and Nyquist fixed to -60 and 0 dB respectively

    # interpolate bins of ap to get value at exact frequencies
    spec_size = ap.shape[1]
    f_spec = np.linspace(0, sr / 2, spec_size)
    interp_f = scipy.interpolate.interp1d(f_spec, ap, kind='linear', axis=-1, assume_sorted=True)

    f_wbap = get_wbap_freqs(n_wbap, with_edges=False)
    wbap = interp_f(f_wbap)

    return wbap


def wbap_to_ap(wbap, spec_size, sr):
    # reconstruct ap by linear interpolation
    n_wbap = wbap.shape[1]
    f_wbap = get_wbap_freqs(n_wbap, with_edges=True, sr=sr)
    a_wbap = np.empty((wbap.shape[0], n_wbap + 2))
    a_wbap[:, 0] = -60.0
    a_wbap[:, 1:-1] = wbap[:, :]
    a_wbap[:, -1] = -wbap_eps  # 1 - 10**((-1e-12)/20) in linear
    interp_f = scipy.interpolate.interp1d(f_wbap, a_wbap, kind='linear', axis=-1, assume_sorted=True)

    f_spec = np.linspace(0, sr / 2, spec_size)
    ap = interp_f(f_spec)

    return ap
def griffinlim(spectrogram, n_iter = 50, window = 'hann', n_fft = 1024, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        # full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = istft(spectrogram,angles)
        rebuilt = stft(inverse)[:spectrogram.shape[0],:]
        angles = np.exp(1j * np.angle(rebuilt))
        progress(i,n_iter)
        # import pdb;pdb.set_trace()

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    # full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = istft(spectrogram, angles)

    return inverse


def shuffle_two(a,b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)
    return a2, b2


def stft(data, window=np.hanning(1024),
         hopsize=256.0, nfft=1024.0, fs=44100.0):
    """
    X, F, N = stft(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
        F                     :
            values of frequencies at each Fourier bins
        N                     :
            central time at the middle of each analysis
            window
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    lengthData = data.size
    
    # should be the number of frames by YAAFE:
    numberFrames = np.ceil(lengthData / np.double(hopsize)) + 2
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = (numberFrames-1) * hopsize + lengthWindow

    # import pdb;pdb.set_trace()
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data

    # import pdb;pdb.set_trace()
    if len(data.shape)>1:
        data = np.mean(data, axis = -1)
    data = np.concatenate((np.zeros(int(lengthWindow/2)), data))
    
    # zero-padding data such that it holds an exact number of frames

    data = np.concatenate((data, np.zeros(int(newLengthData - data.size))))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2 + 1
    
    STFT = np.zeros([int(numberFrames), int(numberFrequencies)], dtype=complex)
    
    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = n*hopsize
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[int(beginFrame):int(endFrame)]
        STFT[int(n),:] = np.fft.rfft(frameToProcess, np.int32(nfft), norm="ortho")
        
    # frequency and time stamps:
    F = np.arange(numberFrequencies)/np.double(nfft)*fs
    N = np.arange(numberFrames)*hopsize/np.double(fs)
    
    return STFT

def istft(mag, phase, window=np.hanning(1024),
         hopsize=256.0, nfft=1024.0, fs=44100.0,
          analysisWindow=None):
    """
    data = istft_norm(X,window=sinebell(2048),hopsize=1024.0,nfft=2048.0,fs=44100)
    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.
    Inputs:
        X                     :
            STFT of the signal, to be \"inverted\"
        window=sinebell(2048) :
            synthesis window
            (should be the \"complementary\" window
            for the analysis window)
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
    Outputs:
        data                  :
            time series corresponding to the given STFT
            the first half-window is removed, complying
            with the STFT computation given in the
            function stft
    """
    X = mag * np.exp(1j*phase)
    X = X.T
    if analysisWindow is None:
        analysisWindow = window

    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize*(numberFrames-1) + lengthWindow)

    normalisationSeq = np.zeros(lengthData)

    data = np.zeros(lengthData)

    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft), norm = 'ortho')
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)

    data = data[int(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.

    data = data / normalisationSeq

    return data

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isinf(y), lambda z: z.nonzero()[0]

def file_to_stft(input_file, mode =0):
    audio,fs=sf.read(input_file)
    if mode == 0 :
        mixture = (audio[:,0]+audio[:,1])*0.7
        mix_stft=abs(stft(mixture))
    
        return mix_stft
    elif mode ==1:
        mixture = audio
        mix_stft=abs(stft(mixture))
    
        return mix_stft
    elif mode ==2:
        mixture = audio[:,0]
        mix_stft=abs(stft(mixture))
        return mix_stft
    elif mode ==3:
        mixture = audio

        mix_stft=stft(mixture)
        return abs(mix_stft), np.angle(mix_stft)







def input_to_feats(input_file, mode=0):
    audio,fs=sf.read(input_file)
    if mode == 0 or mode ==2:
        vocals=np.array(audio[:,1])
    elif mode ==1:
        vocals = audio

    feats = stft_to_feats(vocals,fs)


    # harm_in=mgc_to_sp(harmy, 1025, 0.45)
    # ap_in=mgc_to_sp(apy, 1025, 0.45)


    return feats

def stft_to_feats(vocals, fs, mode=config.comp_mode):
    feats=pw.wav2world(vocals,fs,frame_period=5.80498866)

    ap = feats[2].reshape([feats[1].shape[0],feats[1].shape[1]]).astype(np.float32)
    ap = 10.*np.log10(ap**2)
    harm=10*np.log10(feats[1].reshape([feats[2].shape[0],feats[2].shape[1]]))
    feats=pw.wav2world(vocals,fs,frame_period=5.80498866)

    f0 = feats[0]
    # f0 = pitch.extract_f0_sac(vocals, fs, 0.00580498866)

    y=69+12*np.log2(f0/440)
    # import pdb;pdb.set_trace()
    # y = hertz_to_new_base(f0)
    nans, x= nan_helper(y)
    naners=np.isinf(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # y=[float(x-(min_note-1))/float(max_note-(min_note-1)) for x in y]
    y=np.array(y).reshape([len(y),1])
    guy=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,guy),axis=-1)

    if mode == 'mfsc':
        harmy=sp_to_mfsc(harm,60,0.45)
        apy=sp_to_mfsc(ap,4,0.45)
    elif mode == 'mgc':
        harmy=sp_to_mgc(harm,60,0.45)
        apy=sp_to_mgc(ap,4,0.45)

    # import pdb;pdb.set_trace()


    out_feats=np.concatenate((harmy,apy,y.reshape((-1,2))),axis=1) 

    # harm_in=mgc_to_sp(harmy, 1025, 0.45)
    # ap_in=mgc_to_sp(apy, 1025, 0.45)


    return out_feats

def write_ori_ikala(input_file, filename):
    audio,fs = sf.read(input_file)
    mixture = (audio[:,0]+audio[:,1])*0.7
    vocals = np.array(audio[:,1])
    backing = np.array(audio[:,0])
    sf.write(config.VAL_DIR + filename + '_mixture.wav', mixture, fs)
    sf.write(config.VAL_DIR + filename + '_ori_vocals.wav', vocals, fs)
    sf.write(config.VAL_DIR + filename + '_backing.wav', backing, fs)

def write_ori_med(input_file, filename):
    audio,fs = sf.read(input_file)
    mixture = np.array(audio[:,0])
    vocals = np.array(audio[:,1])
    sf.write(config.VAL_DIR + filename + '_mixture.wav', mixture, fs)
    sf.write(config.VAL_DIR + filename + '_ori_vocals.wav', vocals, fs)

def file_to_sac(input_file):
    audio,fs = sf.read(input_file)
    vocals = np.array(audio[:,1])
    feats=pw.wav2world(vocals,fs,frame_period=5.80498866)

    f0 = feats[0]
    # f0 = pitch.extract_f0_sac(vocals, config.fs, 0.00580498866)
    y=69+12*np.log2(f0/440)
    # y = hertz_to_new_base(f0)
    nans, x= nan_helper(y)
    naners=np.isinf(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # y=[float(x-(min_note-1))/float(max_note-(min_note-1)) for x in y]
    y=np.array(y).reshape([len(y),1])
    guy=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,guy),axis=-1)
    return y

def f0_to_hertz(f0):
    # if f0 == 0:
    #     return 0
    # else:
    f0 = f0-69
    f0 = f0/12
    f0 = 2**f0
    f0 = f0*440
    return f0


def hertz_to_new_base(f0):
    # if f0 == 0:
    #     return 0
    # else:
    return 1200*np.log2(f0/10)

def new_base_to_hertz(f0):
    return 2**(f0*10)/1200

def feats_to_audio(in_feats, filename, fs=config.FS, mode=config.comp_mode):
    harm = in_feats[:,:60]
    ap = in_feats[:,60:-2]
    f0 = in_feats[:,-2:]
    # f0[:,0] = f0[:,0]-69
    # f0[:,0] = f0[:,0]/12
    # f0[:,0] = 2**f0[:,0]
    # f0[:,0] = f0[:,0]*440
    f0[:,0] = f0_to_hertz(f0[:,0])

    f0 = f0[:,0]*(1-f0[:,1])


    if mode == 'mfsc':
        harm = mfsc_to_mgc(harm)
        ap = mfsc_to_mgc(ap)


    harm = mgc_to_sp(harm, 1025, 0.45)
    ap = mgc_to_sp(ap, 1025, 0.45)

    harm = 10**(harm/10)
    ap = 10**(ap/20)

    y=pw.synthesize(f0.astype('double'),harm.astype('double'),ap.astype('double'),fs,config.hoptime)
    sf.write(config.VAL_DIR + filename + '.wav', y, fs)

def feats_to_audio_test(in_feats, filename, fs=config.FS, mode=config.comp_mode):
    harm = in_feats[:,:60]
    ap = in_feats[:,60:-2]
    f0 = in_feats[:,-2:]
    f0[:,0] = f0[:,0]-69
    f0[:,0] = f0[:,0]/12
    f0[:,0] = 2**f0[:,0]
    f0[:,0] = f0[:,0]*440


    f0 = f0[:,0]*(1-f0[:,1])


    if mode == 'mfsc':
        harm = mfsc_to_mgc(harm)
        ap = mfsc_to_mgc(ap)


    harm = mgc_to_sp(harm, 1025, 0.45)
    ap = mgc_to_sp(ap, 1025, 0.45)

    harm = 10**(harm/10)
    ap = 10**(ap/20)

    y=pw.synthesize(f0.astype('double'),harm.astype('double'),ap.astype('double'),fs,config.hoptime)
    sf.write('./medley_resynth_test/'+filename+'.wav',y,fs)
    # return harm, ap, f0

def test(ori, re):
    plt.subplot(211)
    plt.imshow(ori.T,origin='lower',aspect='auto')
    plt.subplot(212)
    plt.imshow(re.T,origin='lower',aspect='auto')
    plt.show()


def generate_overlapadd(allmix,time_context=config.max_phr_len, overlap=config.max_phr_len/2,batch_size=config.batch_size):
    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    input_size = allmix.shape[-1]

    i=0
    start=0  
    while (start + time_context) < allmix.shape[0]:
        i = i + 1
        start = start - overlap + time_context 
    fbatch = np.zeros([int(np.ceil(float(i)/batch_size)),batch_size,time_context,input_size])+1e-10
    
    
    i=0
    start=0  

    while (start + time_context) < allmix.shape[0]:
        fbatch[int(i/batch_size),int(i%batch_size),:,:]=allmix[int(start):int(start+time_context),:]
        i = i + 1 #index for each block
        start = start - overlap + time_context #starting point for each block
    
    return fbatch,i

def overlapadd(fbatch,nchunks,overlap=int(config.max_phr_len/2)):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[1]


    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    

    sep = np.zeros((int(nchunks*(time_context-overlap)+time_context),input_size))

    
    i=0
    start=0 
    while i < nchunks:
        #import pdb;pdb.set_trace()
        s = fbatch[int(i/batch_size),int(i%batch_size),:,:]

        #print s1.shape
        if start==0:
            sep[0:time_context] = s

        else:
            #print start+overlap
            #print start+time_context
            sep[int(start+overlap):int(start+time_context)] = s[overlap:time_context]
            sep[start:int(start+overlap)] = window[overlap:]*sep[start:int(start+overlap)] + window[:overlap]*s[:overlap]
        i = i + 1 #index for each block
        start = int(start - overlap + time_context) #starting point for each block
    return sep  


def normalize(inputs, feat, mode=config.norm_mode_in):
    if mode == "max_min":
        maximus = np.load(config.stat_dir+feat+'_maximus.npy')
        minimus = np.load(config.stat_dir+feat+'_minimus.npy')
        # import pdb;pdb.set_trace()
        outputs = (inputs-minimus)/(maximus-minimus)

    elif mode == "mean":
        means = np.load(config.stat_dir+feat+'_means.npy')
        stds = np.load(config.stat_dir+feat+'_stds.npy')
        outputs = (inputs-means)/stds
    elif mode == "clip":
        outputs = np.clip(inputs, 0.0,1.0)

    return outputs


def list_to_file(in_list,filename):
    filer=open(filename,'w')
    for jj in in_list:
        filer.write(str(jj)+'\n')
    filer.close()

def denormalize(inputs, feat, mode=config.norm_mode_in):
    if mode == "max_min":
        maximus = np.load(config.stat_dir+feat+'_maximus.npy')
        minimus = np.load(config.stat_dir+feat+'_minimus.npy')
        # import pdb;pdb.set_trace()
        outputs = (inputs*(maximus-minimus))+minimus

    elif mode == "mean":
        means = np.load(config.stat_dir+feat+'_means.npy')
        stds = np.load(config.stat_dir+feat+'_stds.npy')
        outputs = (inputs*stds)+means
    return outputs

def query_yes_no(question, default="yes"):
    """
    Copied from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def match_time(feat_list):
    """ 
    Matches the shape across the time dimension of a list of arrays.
    Assumes that the first dimension is in time, preserves the other dimensions
    """
    shapes = [f.shape for f in feat_list]
    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[0] for s in shapes])
        new_list = []
        for i in range(len(feat_list)):
            new_list.append(feat_list[i][:min_time])
        feat_list = new_list
    return feat_list
def main():
    out_feats = input_to_feats(config.wav_dir+'10161_chorus.wav')
    feats_to_audio(out_feats, 'test')
    # test(harmy, 10*np.log10(harm))

    # test_sample = np.random.rand(5170,66)

    # fbatch,i = generate_overlapadd(test_sample)

    # sampled = overlapadd(fbatch,i)

    import pdb;pdb.set_trace()




if __name__ == '__main__':
    main()