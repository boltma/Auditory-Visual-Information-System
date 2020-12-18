import matplotlib.pyplot as plt
import pickle
import numpy as np
# import sounddevice as s

# pth = "dataset/train/061_foam_brick/10/audio_data.pkl"
# img = pickle.load(open(pth, 'rb'))["audio"]
# print(img[:,3].shape)
# a = img[:,1]
# print(a)
# b = np.arange(len(a))
# print(b)
# plot.plot(b, a)
# plot.show()

def spectrogram(sig, Fs=44100, NFFT=1024):
    Specs = np.array([])
    for i in range(4):
        if len(sig[:, i]) < 176400:
            s = np.append(sig[:, i], sig[0:176400-len(sig[:, i]), i])
        else:
            s = sig[0:176400, i]
        Spec, freq, time, Ax = plt.specgram(s, NFFT=NFFT, Fs=Fs)
        Spec = np.log10(Spec) * 10 + 220
        Spec = np.maximum(Spec, 0)
        Spec = np.minimum(Spec, 255)
        if i == 0:
            Specs = Spec
        else:
            Specs = np.dstack((Specs, Spec))
        # plt.show()
    # Specs = Specs.swapaxes(0, 2)
    # print(Specs.shape)
    #print(Specs)
    return np.uint8(Specs)
    
