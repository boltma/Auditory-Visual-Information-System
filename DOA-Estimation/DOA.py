import argparse
import math
import os
import numpy as np
import noisereduce as nr
import librosa
from scipy import signal

c = 343.0
d = 0.2 / np.sqrt(2)
T_NEIGHBOR = 2 * d / c
PI = 3.1415926
NOISE_LEN = 20000
SR_MULTIPLIER = 16
BANDPASS_FILTER = signal.firwin(
    1024, [0.02, 0.3], pass_zero=False)

def read_audio(filepth, num):
    name1 = filepth + str(num) + '_mic1.wav'
    name2 = filepth + str(num) + '_mic2.wav'
    name3 = filepth + str(num) + '_mic3.wav'
    name4 = filepth + str(num) + '_mic4.wav' 
    y1, sr = librosa.load(name1, sr=None, mono=False)
    y2, sr = librosa.load(name2, sr=None, mono=False)
    y3, sr = librosa.load(name3, sr=None, mono=False)
    y4, sr = librosa.load(name4, sr=None, mono=False)
    
    y = np.vstack((y1, y2, y3, y4)).T
    return y, sr

def reduce_noise(ch1, ch2, noise_len):
    ''' 去噪 '''
    ch1_noise = ch1[0:noise_len]
    ch1_dn = nr.reduce_noise(audio_clip=ch1, noise_clip=ch1_noise, n_grad_freq=2,
                             n_grad_time=6, n_fft=8192, win_length=8192,
                             hop_length=128, n_std_thresh=1.5, prop_decrease=1)
    ch2_noise = ch2[0:noise_len]
    ch2_dn = nr.reduce_noise(audio_clip=ch2, noise_clip=ch2_noise, n_grad_freq=2,
                             n_grad_time=6, n_fft=8192, win_length=8192,
                             hop_length=128, n_std_thresh=1.5, prop_decrease=1)
    return ch1_dn, ch2_dn


def resample(ch1, ch2, orig_sr, target_sr):
    ''' 变换采样率 '''
    ch1_new = librosa.resample(ch1, orig_sr, target_sr)
    ch2_new = librosa.resample(ch2, orig_sr, target_sr)
    return ch1_new, ch2_new

def bandpass_filter(ch1, ch2):
    ''' 去除信号的低频、高频分量 '''
    ch1_ft = signal.lfilter(BANDPASS_FILTER, [1.0], ch1)
    ch2_ft = signal.lfilter(BANDPASS_FILTER, [1.0], ch2)
    return ch1_ft, ch2_ft

def gccphat(s, Fs):
    gcc = np.zeros((4,4))
    
    for i in range(0, 4):
        for j in range(0, 4):
            #print("%d %d"%(i, j))
            #ch1_ft, ch2_ft = bandpass_filter(s[:,i], s[:,j])
            ch1_dn, ch2_dn = reduce_noise(s[:,i], s[:,j], NOISE_LEN)
            #Fs_up = Fs * SR_MULTIPLIER
            #ch1_up, ch2_up = resample(ch1_dn, ch2_dn, Fs, Fs_up)
            gcc[i, j], cc = gcc_phat(ch1_dn, ch2_dn, Fs)

    return gcc


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc
def acos(z):
    if abs(z) < 1:
        return math.acos(z)
    else:
        z1 = complex(0, -np.log(complex(z, np.sqrt(z*z - 1))))
        #print(z1)
        return z1
ans = np.zeros(14)
for ii in range(1, 15):
    print(ii)
    s, Fs = read_audio('train/', ii)
    gcc = gccphat(s, Fs)
    tau1 = (gcc[0, 1] + gcc[3, 2]) / 2; # horizontal
    tau2 = (gcc[0, 3] + gcc[1, 2]) / 2; # vertical
    z1 = -tau1 * c / d
    z2 = -tau2 * c / d
    theta1_ori = acos(z1) * 180 / PI 
    theta2_ori = acos(z2) * 180 / PI  

    theta1 = theta1_ori.real
    theta2 = theta2_ori.real

    #flag = 1 when theta1 and theta2 are same direction
    if theta2 < 90:
        flag = abs(theta1 - theta2 - 90) < abs(theta2 + theta1 - 90)
    else:
        flag = abs(theta2 - theta1 - 90) < abs(theta2 + theta1 - 270)

    if abs(theta2 - 90) > 40 or theta2_ori.imag != 0:
        theta = (2 * (theta2 < 90) - 1) * theta1 - 45
    elif abs(theta1 - 90) > 40 or theta1_ori.imag != 0:
        theta = 45 - (2 * (flag ^  (theta2 < 90)) - 1) * theta2
    else:
        theta = (((2 * flag - 1) * theta2 + theta1) / 2) * (2 * (theta2 < 90) - 1)

    theta = theta % 360

    ans[ii - 1] = theta
print(ans)