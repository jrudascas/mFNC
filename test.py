import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def butter_bandpass(self, lowcut, highcut, fs, order=2):
    nyq = 1 / (2 * fs)
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
    b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


img = nib.load('/home/jrudascas/Downloads/drive-download-20170722T213449Z-001/20170721_083501DRCASTELLANOS11s011a1001.nii.gz')
data = img.get_data()
affine = img.get_affine()

print(img.get_header().get_zooms()[:3])
print(data.shape)