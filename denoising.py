import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import numpy as np
import random
import matplotlib.pyplot as plt
import matlab.engine
import numpy as np


eng = matlab.engine.start_matlab()


def read(filepath):
    rf = eng.RPread(filepath)
    rf_np = np.array(rf._data)
    rf_np = np.reshape(rf_np, rf.size, order="F")
    return rf_np


class Denoise_us(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, low=6, high=-60, fs=4e7):
        super(Denoise_us, self).__init__()

        samplingFrequency = fs
        samplingInterval = 1 / samplingFrequency

        tpCount = 256
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / samplingFrequency
        frequencies = values / timePeriod

        desired_filter = np.zeros(tpCount // 2 + 1)
        desired_filter[low:high] = 1

        bands = np.append(frequencies, frequencies[-1] + frequencies[1])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel = signal.firwin2(151, bands, desired_filter, fs=fs)[:, np.newaxis, np.newaxis]

    def forward(self, x):
        x = signal.convolve(x, self.kernel, mode='same')
        return x
