import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from scipy.signal import hilbert
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
# from prepare_data import *
from netmodel import *
from median_pool import *
from read_rf import read
import statistics
import time

LR = 1e-5
us_images = [int(500*(3)), int(500*(3)), int(500*(3))]
epoch_list = [500, 500, 500]
repetition = 8
patch_width = [3, 3, 3]
START_pixel = [500, 800, 1100]


def extract_all_patchess(filepath, num, Depth, Start):
    volume = read(filepath)

    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    depth, channels, frames = volume.shape

    if channels == 512:
        volume = volume[:, ::2, :]

    start_depth = Start
    patch_size = 200
    jump = 100

    patches = []
    depth_list = []
    flag = True

    frame_counter = 0
    depth_counter = 0

    while flag:
        for jj in range(9):
            patches.append(volume[start_depth + depth_counter * jump:start_depth + patch_size + depth_counter * jump,
                           10 + 26 * jj:36 + 26 * jj, frame_counter])
            depth_list.append(depth_counter)

        depth_counter += 1
        if depth_counter == Depth:
            frame_counter += 1
            depth_counter = 0
            # patches.pop()
            # depth_list.pop()

        if start_depth + patch_size + depth_counter * jump >= depth:
            frame_counter += 1
            depth_counter = 0
            patches.pop()
            depth_list.pop()

        if frame_counter == frames:
            flag = False

    return np.array(patches), np.array(depth_list)


# Figure 6 in Zone Training Paper Arxiv version
def far_field_3patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-100-100: focus-100+100, 10:36, :], volume[focus-100: focus+100, 10:36, :],
                            volume[focus+100-100: focus+100+100, 10:36, :],
                            volume[focus-100-100: focus-100+100, 36:62, :], volume[focus-100: focus+100, 36:62, :],
                            volume[focus+100-100: focus+100+100, 36:62, :],
                            volume[focus-100-100: focus-100+100, 62:88, :], volume[focus-100: focus+100, 62:88, :],
                            volume[focus+100-100: focus+100+100, 62:88, :],
                            volume[focus-100-100: focus-100+100, 88:114, :], volume[focus-100: focus+100, 88:114, :],
                            volume[focus+100-100: focus+100+100, 88:114, :],
                            volume[focus-100-100: focus-100+100, 114:140, :], volume[focus-100: focus+100, 114:140, :],
                            volume[focus+100-100: focus+100+100, 114:140, :],
                            volume[focus-100-100: focus-100+100, 140:166, :], volume[focus-100: focus+100, 140:166, :],
                            volume[focus+100-100: focus+100+100, 140:166, :],
                            volume[focus-100-100: focus-100+100, 166:192, :], volume[focus-100: focus+100, 166:192, :],
                            volume[focus+100-100: focus+100+100, 166:192, :],
                            volume[focus-100-100: focus-100+100, 192:218, :], volume[focus-100: focus+100, 192:218, :],
                            volume[focus+100-100: focus+100+100, 192:218, :],
                            volume[focus-100-100: focus-100+100, 218:244, :], volume[focus-100: focus+100, 218:244, :],
                            volume[focus+100-100: focus+100+100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_4patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-150-100: focus-150+100, 10:36, :], volume[focus-50-100: focus-50+100, 10:36, :],
                            volume[focus+50-100: focus+50+100, 10:36, :], volume[focus+150-100: focus+150+100, 10:36, :],
                            volume[focus-150-100: focus-150+100, 36:62, :], volume[focus-50-100: focus-50+100, 36:62, :],
                            volume[focus+50-100: focus+50+100, 36:62, :], volume[focus+150-100: focus+150+100, 36:62, :],
                            volume[focus-150-100: focus-150+100, 62:88, :], volume[focus-50-100: focus-50+100, 62:88, :],
                            volume[focus+50-100: focus+50+100, 62:88, :], volume[focus+150-100: focus+150+100, 62:88, :],
                            volume[focus-150-100: focus-150+100, 88:114, :], volume[focus-50-100: focus-50+100, 88:114, :],
                            volume[focus+50-100: focus+50+100, 88:114, :], volume[focus+150-100: focus+150+100, 88:114, :],
                            volume[focus-150-100: focus-150+100, 114:140, :], volume[focus-50-100: focus-50+100, 114:140, :],
                            volume[focus+50-100: focus+50+100, 114:140, :], volume[focus+150-100: focus+150+100, 114:140, :],
                            volume[focus-150-100: focus-150+100, 140:166, :], volume[focus-50-100: focus-50+100, 140:166, :],
                            volume[focus+50-100: focus+50+100, 140:166, :], volume[focus+150-100: focus+150+100, 140:166, :],
                            volume[focus-150-100: focus-150+100, 166:192, :], volume[focus-50-100: focus-50+100, 166:192, :],
                            volume[focus+50-100: focus+50+100, 166:192, :], volume[focus+150-100: focus+150+100, 166:192, :],
                            volume[focus-150-100: focus-150+100, 192:218, :], volume[focus-50-100: focus-50+100, 192:218, :],
                            volume[focus+50-100: focus+50+100, 192:218, :], volume[focus+150-100: focus+150+100, 192:218, :],
                            volume[focus-150-100: focus-150+100, 218:244, :], volume[focus-50-100: focus-50+100, 218:244, :],
                            volume[focus+50-100: focus+50+100, 218:244, :], volume[focus+150-100: focus+150+100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_5patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-100-100: focus-100+100, 10:36, :], volume[focus-100: focus+100, 10:36, :],
                            volume[focus+100-100: focus+100+100, 10:36, :],
                            volume[focus+200-100: focus+200+100, 10:36, :], volume[focus-200-100: focus-200+100, 10:36, :],
                            volume[focus-100-100: focus-100+100, 36:62, :], volume[focus-100: focus+100, 36:62, :],
                            volume[focus+100-100: focus+100+100, 36:62, :],
                            volume[focus+200-100: focus+200+100, 36:62, :],
                            volume[focus-200-100: focus-200+100, 36:62, :],
                            volume[focus-100-100: focus-100+100, 62:88, :], volume[focus-100: focus+100, 62:88, :],
                            volume[focus+100-100: focus+100+100, 62:88, :],
                            volume[focus+200-100: focus+200+100, 62:88, :],
                            volume[focus-200-100: focus-200+100, 62:88, :],
                            volume[focus-100-100: focus-100+100, 88:114, :], volume[focus-100: focus+100, 88:114, :],
                            volume[focus+100-100: focus+100+100, 88:114, :],
                            volume[focus+200-100: focus+200+100, 88:114, :],
                            volume[focus-200-100: focus-200+100, 88:114, :],
                            volume[focus-100-100: focus-100+100, 114:140, :], volume[focus-100: focus+100, 114:140, :],
                            volume[focus+100-100: focus+100+100, 114:140, :],
                            volume[focus+200-100: focus+200+100, 114:140, :],
                            volume[focus-200-100: focus-200+100, 114:140, :],
                            volume[focus-100-100: focus-100+100, 140:166, :], volume[focus-100: focus+100, 140:166, :],
                            volume[focus+100-100: focus+100+100, 140:166, :],
                            volume[focus+200-100: focus+200+100, 140:166, :],
                            volume[focus-200-100: focus-200+100, 140:166, :],
                            volume[focus-100-100: focus-100+100, 166:192, :], volume[focus-100: focus+100, 166:192, :],
                            volume[focus+100-100: focus+100+100, 166:192, :],
                            volume[focus+200-100: focus+200+100, 166:192, :],
                            volume[focus-200-100: focus-200+100, 166:192, :],
                            volume[focus-100-100: focus-100+100, 192:218, :], volume[focus-100: focus+100, 192:218, :],
                            volume[focus+100-100: focus+100+100, 192:218, :],
                            volume[focus+200-100: focus+200+100, 192:218, :],
                            volume[focus-200-100: focus-200+100, 192:218, :],
                            volume[focus-100-100: focus-100+100, 218:244, :], volume[focus-100: focus+100, 218:244, :],
                            volume[focus+100-100: focus+100+100, 218:244, :],
                            volume[focus+200-100: focus+200+100, 218:244, :],
                            volume[focus-200-100: focus-200+100, 218:244, :],), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_6patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-150-100: focus-150+100, 10:36, :], volume[focus-50-100: focus-50+100, 10:36, :],
                            volume[focus+50-100: focus+50+100, 10:36, :], volume[focus+150-100: focus+150+100, 10:36, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 10:36, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 10:36, :],
                            volume[focus-150-100: focus-150+100, 36:62, :], volume[focus-50-100: focus-50+100, 36:62, :],
                            volume[focus+50-100: focus+50+100, 36:62, :], volume[focus+150-100: focus+150+100, 36:62, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 36:62, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 36:62, :],
                            volume[focus-150-100: focus-150+100, 62:88, :], volume[focus-50-100: focus-50+100, 62:88, :],
                            volume[focus+50-100: focus+50+100, 62:88, :], volume[focus+150-100: focus+150+100, 62:88, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 62:88, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 62:88, :],
                            volume[focus-150-100: focus-150+100, 88:114, :], volume[focus-50-100: focus-50+100, 88:114, :],
                            volume[focus+50-100: focus+50+100, 88:114, :], volume[focus+150-100: focus+150+100, 88:114, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 88:114, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 88:114, :],
                            volume[focus-150-100: focus-150+100, 114:140, :], volume[focus-50-100: focus-50+100, 114:140, :],
                            volume[focus+50-100: focus+50+100, 114:140, :], volume[focus+150-100: focus+150+100, 114:140, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 114:140, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 114:140, :],
                            volume[focus-150-100: focus-150+100, 140:166, :], volume[focus-50-100: focus-50+100, 140:166, :],
                            volume[focus+50-100: focus+50+100, 140:166, :], volume[focus+150-100: focus+150+100, 140:166, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 140:166, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 140:166, :],
                            volume[focus-150-100: focus-150+100, 166:192, :], volume[focus-50-100: focus-50+100, 166:192, :],
                            volume[focus+50-100: focus+50+100, 166:192, :], volume[focus+150-100: focus+150+100, 166:192, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 166:192, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 166:192, :],
                            volume[focus-150-100: focus-150+100, 192:218, :], volume[focus-50-100: focus-50+100, 192:218, :],
                            volume[focus+50-100: focus+50+100, 192:218, :], volume[focus+150-100: focus+150+100, 192:218, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 192:218, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 192:218, :],
                            volume[focus-150-100: focus-150+100, 218:244, :], volume[focus-50-100: focus-50+100, 218:244, :],
                            volume[focus+50-100: focus+50+100, 218:244, :], volume[focus+150-100: focus+150+100, 218:244, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 218:244, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_7patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-100-100: focus-100+100, 10:36, :], volume[focus-100: focus+100, 10:36, :],
                            volume[focus+100-100: focus+100+100, 10:36, :],
                            volume[focus+200-100: focus+200+100, 10:36, :], volume[focus-200-100: focus-200+100, 10:36, :],
                            volume[focus+300-100: focus+300+100, 10:36, :], volume[focus-300-100: focus-300+100, 10:36, :],
                            volume[focus-100-100: focus-100+100, 36:62, :], volume[focus-100: focus+100, 36:62, :],
                            volume[focus+100-100: focus+100+100, 36:62, :],
                            volume[focus+200-100: focus+200+100, 36:62, :],
                            volume[focus-200-100: focus-200+100, 36:62, :],
                            volume[focus+300-100: focus+300+100, 36:62, :],
                            volume[focus-300-100: focus-300+100, 36:62, :],
                            volume[focus-100-100: focus-100+100, 62:88, :], volume[focus-100: focus+100, 62:88, :],
                            volume[focus+100-100: focus+100+100, 62:88, :],
                            volume[focus+200-100: focus+200+100, 62:88, :],
                            volume[focus-200-100: focus-200+100, 62:88, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 62:88, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 62:88, :],
                            volume[focus-100-100: focus-100+100, 88:114, :], volume[focus-100: focus+100, 88:114, :],
                            volume[focus+100-100: focus+100+100, 88:114, :],
                            volume[focus+200-100: focus+200+100, 88:114, :],
                            volume[focus-200-100: focus-200+100, 88:114, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 88:114, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 88:114, :],
                            volume[focus-100-100: focus-100+100, 114:140, :], volume[focus-100: focus+100, 114:140, :],
                            volume[focus+100-100: focus+100+100, 114:140, :],
                            volume[focus+200-100: focus+200+100, 114:140, :],
                            volume[focus-200-100: focus-200+100, 114:140, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 114:140, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 114:140, :],
                            volume[focus-100-100: focus-100+100, 140:166, :], volume[focus-100: focus+100, 140:166, :],
                            volume[focus+100-100: focus+100+100, 140:166, :],
                            volume[focus+200-100: focus+200+100, 140:166, :],
                            volume[focus-200-100: focus-200+100, 140:166, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 140:166, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 140:166, :],
                            volume[focus-100-100: focus-100+100, 166:192, :], volume[focus-100: focus+100, 166:192, :],
                            volume[focus+100-100: focus+100+100, 166:192, :],
                            volume[focus+200-100: focus+200+100, 166:192, :],
                            volume[focus-200-100: focus-200+100, 166:192, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 166:192, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 166:192, :],
                            volume[focus-100-100: focus-100+100, 192:218, :], volume[focus-100: focus+100, 192:218, :],
                            volume[focus+100-100: focus+100+100, 192:218, :],
                            volume[focus+200-100: focus+200+100, 192:218, :],
                            volume[focus-200-100: focus-200+100, 192:218, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 192:218, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 192:218, :],
                            volume[focus-100-100: focus-100+100, 218:244, :], volume[focus-100: focus+100, 218:244, :],
                            volume[focus+100-100: focus+100+100, 218:244, :],
                            volume[focus+200-100: focus+200+100, 218:244, :],
                            volume[focus-200-100: focus-200+100, 218:244, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 218:244, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_8patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-150-100: focus-150+100, 10:36, :], volume[focus-50-100: focus-50+100, 10:36, :],
                            volume[focus+50-100: focus+50+100, 10:36, :], volume[focus+150-100: focus+150+100, 10:36, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 10:36, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 10:36, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 10:36, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 10:36, :],
                            volume[focus-150-100: focus-150+100, 36:62, :], volume[focus-50-100: focus-50+100, 36:62, :],
                            volume[focus+50-100: focus+50+100, 36:62, :], volume[focus+150-100: focus+150+100, 36:62, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 36:62, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 36:62, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 36:62, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 36:62, :],
                            volume[focus-150-100: focus-150+100, 62:88, :], volume[focus-50-100: focus-50+100, 62:88, :],
                            volume[focus+50-100: focus+50+100, 62:88, :], volume[focus+150-100: focus+150+100, 62:88, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 62:88, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 62:88, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 62:88, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 62:88, :],
                            volume[focus-150-100: focus-150+100, 88:114, :], volume[focus-50-100: focus-50+100, 88:114, :],
                            volume[focus+50-100: focus+50+100, 88:114, :], volume[focus+150-100: focus+150+100, 88:114, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 88:114, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 88:114, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 88:114, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 88:114, :],
                            volume[focus-150-100: focus-150+100, 114:140, :], volume[focus-50-100: focus-50+100, 114:140, :],
                            volume[focus+50-100: focus+50+100, 114:140, :], volume[focus+150-100: focus+150+100, 114:140, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 114:140, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 114:140, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 114:140, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 114:140, :],
                            volume[focus-150-100: focus-150+100, 140:166, :], volume[focus-50-100: focus-50+100, 140:166, :],
                            volume[focus+50-100: focus+50+100, 140:166, :], volume[focus+150-100: focus+150+100, 140:166, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 140:166, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 140:166, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 140:166, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 140:166, :],
                            volume[focus-150-100: focus-150+100, 166:192, :], volume[focus-50-100: focus-50+100, 166:192, :],
                            volume[focus+50-100: focus+50+100, 166:192, :], volume[focus+150-100: focus+150+100, 166:192, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 166:192, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 166:192, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 166:192, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 166:192, :],
                            volume[focus-150-100: focus-150+100, 192:218, :], volume[focus-50-100: focus-50+100, 192:218, :],
                            volume[focus+50-100: focus+50+100, 192:218, :], volume[focus+150-100: focus+150+100, 192:218, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 192:218, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 192:218, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 192:218, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 192:218, :],
                            volume[focus-150-100: focus-150+100, 218:244, :], volume[focus-50-100: focus-50+100, 218:244, :],
                            volume[focus+50-100: focus+50+100, 218:244, :], volume[focus+150-100: focus+150+100, 218:244, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 218:244, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 218:244, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 218:244, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_9patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-100-100: focus-100+100, 10:36, :], volume[focus-100: focus+100, 10:36, :],
                            volume[focus+100-100: focus+100+100, 10:36, :],
                            volume[focus+200-100: focus+200+100, 10:36, :], volume[focus-200-100: focus-200+100, 10:36, :],
                            volume[focus+300-100: focus+300+100, 10:36, :], volume[focus-300-100: focus-300+100, 10:36, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 10:36, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 10:36, :],
                              volume[focus-100-100: focus-100+100, 36:62, :], volume[focus-100: focus+100, 36:62, :],
                            volume[focus+100-100: focus+100+100, 36:62, :],
                            volume[focus+200-100: focus+200+100, 36:62, :],
                            volume[focus-200-100: focus-200+100, 36:62, :],
                            volume[focus+300-100: focus+300+100, 36:62, :],
                            volume[focus-300-100: focus-300+100, 36:62, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 36:62, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 36:62, :],
                            volume[focus-100-100: focus-100+100, 62:88, :], volume[focus-100: focus+100, 62:88, :],
                            volume[focus+100-100: focus+100+100, 62:88, :],
                            volume[focus+200-100: focus+200+100, 62:88, :],
                            volume[focus-200-100: focus-200+100, 62:88, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 62:88, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 62:88, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 62:88, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 62:88, :],
                            volume[focus-100-100: focus-100+100, 88:114, :], volume[focus-100: focus+100, 88:114, :],
                            volume[focus+100-100: focus+100+100, 88:114, :],
                            volume[focus+200-100: focus+200+100, 88:114, :],
                            volume[focus-200-100: focus-200+100, 88:114, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 88:114, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 88:114, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 88:114, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 88:114, :],
                            volume[focus-100-100: focus-100+100, 114:140, :], volume[focus-100: focus+100, 114:140, :],
                            volume[focus+100-100: focus+100+100, 114:140, :],
                            volume[focus+200-100: focus+200+100, 114:140, :],
                            volume[focus-200-100: focus-200+100, 114:140, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 114:140, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 114:140, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 114:140, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 114:140, :],
                            volume[focus-100-100: focus-100+100, 140:166, :], volume[focus-100: focus+100, 140:166, :],
                            volume[focus+100-100: focus+100+100, 140:166, :],
                            volume[focus+200-100: focus+200+100, 140:166, :],
                            volume[focus-200-100: focus-200+100, 140:166, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 140:166, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 140:166, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 140:166, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 140:166, :],
                            volume[focus-100-100: focus-100+100, 166:192, :], volume[focus-100: focus+100, 166:192, :],
                            volume[focus+100-100: focus+100+100, 166:192, :],
                            volume[focus+200-100: focus+200+100, 166:192, :],
                            volume[focus-200-100: focus-200+100, 166:192, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 166:192, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 166:192, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 166:192, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 166:192, :],
                            volume[focus-100-100: focus-100+100, 192:218, :], volume[focus-100: focus+100, 192:218, :],
                            volume[focus+100-100: focus+100+100, 192:218, :],
                            volume[focus+200-100: focus+200+100, 192:218, :],
                            volume[focus-200-100: focus-200+100, 192:218, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 192:218, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 192:218, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 192:218, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 192:218, :],
                            volume[focus-100-100: focus-100+100, 218:244, :], volume[focus-100: focus+100, 218:244, :],
                            volume[focus+100-100: focus+100+100, 218:244, :],
                            volume[focus+200-100: focus+200+100, 218:244, :],
                            volume[focus-200-100: focus-200+100, 218:244, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 218:244, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 218:244, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 218:244, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_10patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-150-100: focus-150+100, 10:36, :], volume[focus-50-100: focus-50+100, 10:36, :],
                            volume[focus+50-100: focus+50+100, 10:36, :], volume[focus+150-100: focus+150+100, 10:36, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 10:36, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 10:36, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 10:36, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 10:36, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 10:36, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 10:36, :],
                            volume[focus-150-100: focus-150+100, 36:62, :], volume[focus-50-100: focus-50+100, 36:62, :],
                            volume[focus+50-100: focus+50+100, 36:62, :], volume[focus+150-100: focus+150+100, 36:62, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 36:62, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 36:62, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 36:62, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 36:62, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 36:62, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 36:62, :],
                            volume[focus-150-100: focus-150+100, 62:88, :], volume[focus-50-100: focus-50+100, 62:88, :],
                            volume[focus+50-100: focus+50+100, 62:88, :], volume[focus+150-100: focus+150+100, 62:88, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 62:88, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 62:88, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 62:88, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 62:88, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 62:88, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 62:88, :],
                            volume[focus-150-100: focus-150+100, 88:114, :], volume[focus-50-100: focus-50+100, 88:114, :],
                            volume[focus+50-100: focus+50+100, 88:114, :], volume[focus+150-100: focus+150+100, 88:114, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 88:114, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 88:114, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 88:114, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 88:114, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 88:114, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 88:114, :],
                            volume[focus-150-100: focus-150+100, 114:140, :], volume[focus-50-100: focus-50+100, 114:140, :],
                            volume[focus+50-100: focus+50+100, 114:140, :], volume[focus+150-100: focus+150+100, 114:140, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 114:140, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 114:140, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 114:140, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 114:140, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 114:140, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 114:140, :],
                            volume[focus-150-100: focus-150+100, 140:166, :], volume[focus-50-100: focus-50+100, 140:166, :],
                            volume[focus+50-100: focus+50+100, 140:166, :], volume[focus+150-100: focus+150+100, 140:166, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 140:166, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 140:166, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 140:166, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 140:166, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 140:166, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 140:166, :],
                            volume[focus-150-100: focus-150+100, 166:192, :], volume[focus-50-100: focus-50+100, 166:192, :],
                            volume[focus+50-100: focus+50+100, 166:192, :], volume[focus+150-100: focus+150+100, 166:192, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 166:192, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 166:192, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 166:192, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 166:192, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 166:192, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 166:192, :],
                            volume[focus-150-100: focus-150+100, 192:218, :], volume[focus-50-100: focus-50+100, 192:218, :],
                            volume[focus+50-100: focus+50+100, 192:218, :], volume[focus+150-100: focus+150+100, 192:218, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 192:218, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 192:218, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 192:218, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 192:218, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 192:218, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 192:218, :],
                            volume[focus-150-100: focus-150+100, 218:244, :], volume[focus-50-100: focus-50+100, 218:244, :],
                            volume[focus+50-100: focus+50+100, 218:244, :], volume[focus+150-100: focus+150+100, 218:244, :],
                            volume[focus + 250 - 100: focus + 250 + 100, 218:244, :],
                            volume[focus - 250 - 100: focus - 250 + 100, 218:244, :],
                            volume[focus + 350 - 100: focus + 350 + 100, 218:244, :],
                            volume[focus - 350 - 100: focus - 350 + 100, 218:244, :],
                            volume[focus + 450 - 100: focus + 450 + 100, 218:244, :],
                            volume[focus - 450 - 100: focus - 450 + 100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_11patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-100-100: focus-100+100, 10:36, :], volume[focus-100: focus+100, 10:36, :],
                            volume[focus+100-100: focus+100+100, 10:36, :],
                            volume[focus+200-100: focus+200+100, 10:36, :], volume[focus-200-100: focus-200+100, 10:36, :],
                            volume[focus+300-100: focus+300+100, 10:36, :], volume[focus-300-100: focus-300+100, 10:36, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 10:36, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 10:36, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 10:36, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 10:36, :],
                            volume[focus-100-100: focus-100+100, 36:62, :], volume[focus-100: focus+100, 36:62, :],
                            volume[focus+100-100: focus+100+100, 36:62, :],
                            volume[focus+200-100: focus+200+100, 36:62, :],
                            volume[focus-200-100: focus-200+100, 36:62, :],
                            volume[focus+300-100: focus+300+100, 36:62, :],
                            volume[focus-300-100: focus-300+100, 36:62, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 36:62, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 36:62, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 36:62, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 36:62, :],
                            volume[focus-100-100: focus-100+100, 62:88, :], volume[focus-100: focus+100, 62:88, :],
                            volume[focus+100-100: focus+100+100, 62:88, :],
                            volume[focus+200-100: focus+200+100, 62:88, :],
                            volume[focus-200-100: focus-200+100, 62:88, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 62:88, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 62:88, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 62:88, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 62:88, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 62:88, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 62:88, :],
                            volume[focus-100-100: focus-100+100, 88:114, :], volume[focus-100: focus+100, 88:114, :],
                            volume[focus+100-100: focus+100+100, 88:114, :],
                            volume[focus+200-100: focus+200+100, 88:114, :],
                            volume[focus-200-100: focus-200+100, 88:114, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 88:114, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 88:114, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 88:114, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 88:114, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 88:114, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 88:114, :],
                            volume[focus-100-100: focus-100+100, 114:140, :], volume[focus-100: focus+100, 114:140, :],
                            volume[focus+100-100: focus+100+100, 114:140, :],
                            volume[focus+200-100: focus+200+100, 114:140, :],
                            volume[focus-200-100: focus-200+100, 114:140, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 114:140, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 114:140, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 114:140, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 114:140, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 114:140, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 114:140, :],
                            volume[focus-100-100: focus-100+100, 140:166, :], volume[focus-100: focus+100, 140:166, :],
                            volume[focus+100-100: focus+100+100, 140:166, :],
                            volume[focus+200-100: focus+200+100, 140:166, :],
                            volume[focus-200-100: focus-200+100, 140:166, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 140:166, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 140:166, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 140:166, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 140:166, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 140:166, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 140:166, :],
                            volume[focus-100-100: focus-100+100, 166:192, :], volume[focus-100: focus+100, 166:192, :],
                            volume[focus+100-100: focus+100+100, 166:192, :],
                            volume[focus+200-100: focus+200+100, 166:192, :],
                            volume[focus-200-100: focus-200+100, 166:192, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 166:192, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 166:192, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 166:192, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 166:192, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 166:192, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 166:192, :],
                            volume[focus-100-100: focus-100+100, 192:218, :], volume[focus-100: focus+100, 192:218, :],
                            volume[focus+100-100: focus+100+100, 192:218, :],
                            volume[focus+200-100: focus+200+100, 192:218, :],
                            volume[focus-200-100: focus-200+100, 192:218, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 192:218, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 192:218, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 192:218, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 192:218, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 192:218, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 192:218, :],
                            volume[focus-100-100: focus-100+100, 218:244, :], volume[focus-100: focus+100, 218:244, :],
                            volume[focus+100-100: focus+100+100, 218:244, :],
                            volume[focus+200-100: focus+200+100, 218:244, :],
                            volume[focus-200-100: focus-200+100, 218:244, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 218:244, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 218:244, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 218:244, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 218:244, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 218:244, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 218:244, :]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def far_field_13patches(volume, focus):

    focus = int(focus)
    patches = np.concatenate((volume[focus-100-100: focus-100+100, 10:36, :], volume[focus-100: focus+100, 10:36, :],
                            volume[focus+100-100: focus+100+100, 10:36, :],
                            volume[focus+200-100: focus+200+100, 10:36, :], volume[focus-200-100: focus-200+100, 10:36, :],
                            volume[focus+300-100: focus+300+100, 10:36, :], volume[focus-300-100: focus-300+100, 10:36, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 10:36, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 10:36, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 10:36, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 10:36, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 10:36, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 10:36, :],
                            volume[focus-100-100: focus-100+100, 36:62, :], volume[focus-100: focus+100, 36:62, :],
                            volume[focus+100-100: focus+100+100, 36:62, :],
                            volume[focus+200-100: focus+200+100, 36:62, :],
                            volume[focus-200-100: focus-200+100, 36:62, :],
                            volume[focus+300-100: focus+300+100, 36:62, :],
                            volume[focus-300-100: focus-300+100, 36:62, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 36:62, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 36:62, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 36:62, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 36:62, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 36:62, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 36:62, :],
                            volume[focus-100-100: focus-100+100, 62:88, :], volume[focus-100: focus+100, 62:88, :],
                            volume[focus+100-100: focus+100+100, 62:88, :],
                            volume[focus+200-100: focus+200+100, 62:88, :],
                            volume[focus-200-100: focus-200+100, 62:88, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 62:88, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 62:88, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 62:88, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 62:88, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 62:88, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 62:88, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 62:88, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 62:88, :],
                            volume[focus-100-100: focus-100+100, 88:114, :], volume[focus-100: focus+100, 88:114, :],
                            volume[focus+100-100: focus+100+100, 88:114, :],
                            volume[focus+200-100: focus+200+100, 88:114, :],
                            volume[focus-200-100: focus-200+100, 88:114, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 88:114, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 88:114, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 88:114, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 88:114, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 88:114, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 88:114, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 88:114, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 88:114, :],
                            volume[focus-100-100: focus-100+100, 114:140, :], volume[focus-100: focus+100, 114:140, :],
                            volume[focus+100-100: focus+100+100, 114:140, :],
                            volume[focus+200-100: focus+200+100, 114:140, :],
                            volume[focus-200-100: focus-200+100, 114:140, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 114:140, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 114:140, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 114:140, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 114:140, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 114:140, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 114:140, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 114:140, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 114:140, :],
                            volume[focus-100-100: focus-100+100, 140:166, :], volume[focus-100: focus+100, 140:166, :],
                            volume[focus+100-100: focus+100+100, 140:166, :],
                            volume[focus+200-100: focus+200+100, 140:166, :],
                            volume[focus-200-100: focus-200+100, 140:166, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 140:166, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 140:166, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 140:166, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 140:166, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 140:166, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 140:166, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 140:166, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 140:166, :],
                            volume[focus-100-100: focus-100+100, 166:192, :], volume[focus-100: focus+100, 166:192, :],
                            volume[focus+100-100: focus+100+100, 166:192, :],
                            volume[focus+200-100: focus+200+100, 166:192, :],
                            volume[focus-200-100: focus-200+100, 166:192, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 166:192, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 166:192, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 166:192, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 166:192, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 166:192, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 166:192, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 166:192, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 166:192, :],
                            volume[focus-100-100: focus-100+100, 192:218, :], volume[focus-100: focus+100, 192:218, :],
                            volume[focus+100-100: focus+100+100, 192:218, :],
                            volume[focus+200-100: focus+200+100, 192:218, :],
                            volume[focus-200-100: focus-200+100, 192:218, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 192:218, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 192:218, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 192:218, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 192:218, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 192:218, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 192:218, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 192:218, :],
                            volume[focus - 600 - 100: focus - 600 + 100, 192:218, :],
                            volume[focus-100-100: focus-100+100, 218:244, :], volume[focus-100: focus+100, 218:244, :],
                            volume[focus+100-100: focus+100+100, 218:244, :],
                            volume[focus+200-100: focus+200+100, 218:244, :],
                            volume[focus-200-100: focus-200+100, 218:244, :],
                            volume[focus + 300 - 100: focus + 300 + 100, 218:244, :],
                            volume[focus - 300 - 100: focus - 300 + 100, 218:244, :],
                            volume[focus + 400 - 100: focus + 400 + 100, 218:244, :],
                            volume[focus - 400 - 100: focus - 400 + 100, 218:244, :],
                            volume[focus + 500 - 100: focus + 500 + 100, 218:244, :],
                            volume[focus - 500 - 100: focus - 500 + 100, 218:244, :],
                            volume[focus + 600 - 100: focus + 600 + 100, 218:244,:],
                            volume[focus - 600 - 100: focus - 600 + 100, 218:244,:]), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)

    return patches


def test_function(x_test, y_test, mean_data, std_data, PATH):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Using {} device".format(device))

    net = AlexNet_review(3).to(device)
    #print(net)
    # parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters:{parameter_number}")

    net.load_state_dict(torch.load(PATH))
    x_test = (x_test - mean_data) / std_data

    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to("cuda")
    y_test_gpu = torch.from_numpy(y_test).long().to("cuda")

    dataset = TensorDataset(x_test_gpu, y_test_gpu)
    test_loader = DataLoader(dataset, batch_size=64*2, pin_memory=False, shuffle=True)

    # prepare to count predictions for each class
    classes = ["phantom1", "phantom2", "phantom3"]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    data_matrix = np.zeros((3, 3))
    net.eval()

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                if label == 0:
                    data_matrix[0, prediction] += 1
                elif label == 1:
                    data_matrix[1, prediction] += 1
                elif label == 2:
                    data_matrix[2, prediction] += 1

    # print accuracy for each class
    total = 0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        total = total + accuracy
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

    print("Average Accuracy is: {:.1f} %".format(total/3))
    return total/3


def train_function(x_train, x_valid, x_test, y_train, y_valid, y_test, PATH, epoch_num, LR):
    accuracies = []

    # # Calculate Mean
    mean_data = np.mean(x_train, axis=0)
    std_data = np.std(x_train, axis=0)

    x_valid = (x_valid - mean_data) / std_data
    x_train = (x_train - mean_data) / std_data
    x_test = (x_test - mean_data) / std_data

    x_train_gpu = torch.from_numpy(x_train[:, np.newaxis, :, :]).float().to("cuda")
    y_train_gpu = torch.from_numpy(y_train).long().to("cuda")
    x_valid_gpu = torch.from_numpy(x_valid[:, np.newaxis, :, :]).float().to("cuda")
    y_valid_gpu = torch.from_numpy(y_valid).long().to("cuda")
    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to("cuda")
    y_test_gpu = torch.from_numpy(y_test).long().to("cuda")

    dataset = TensorDataset(x_train_gpu, y_train_gpu)
    train_loader = DataLoader(dataset, batch_size=64*2, pin_memory=False, shuffle=True)

    dataset = TensorDataset(x_valid_gpu, y_valid_gpu)
    valid_loader = DataLoader(dataset, batch_size=64*2, pin_memory=False, shuffle=True)

    dataset = TensorDataset(x_test_gpu, y_test_gpu)
    test_loader = DataLoader(dataset, batch_size=64*2, pin_memory=False, shuffle=True)

    # for X, y in train_loader:
    #     print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Using {} device".format(device))

    # net = MNet().to(device)
    net = AlexNet_review(3).float().to(device)
    # print(net)
    parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters:{parameter_number}")

    criterion = nn.CrossEntropyLoss(torch.tensor([1, 1, 1]).float().to(device))
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_epoch = []
    validation_acc = []
    test_acc = []

    hflipper = T.RandomHorizontalFlip(p=0.5)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs = inputs.float()
            # labels = labels.long()
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Data Augmentation  https://pytorch.org/vision/stable/transforms.html
            inputs = hflipper(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            # print statistics
            running_loss += loss.item()
        loss_epoch.append(running_loss)

        print('[%d] loss: %.3f' % (epoch + 1, running_loss))
        if (epoch + 1) % 100 == 0:
            classes = ["phantom1", "phantom2", "phantom3"]
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            data_matrix = np.zeros((3, 3))
            net.eval()

            # again no gradients needed
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    # inputs = inputs.float()
                    # labels = labels.long()
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    outputs = net(inputs)
                    _, predictions = torch.max(outputs, 1)
                    # collect the correct predictions for each class
                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
                        if label == 0:
                            data_matrix[0, prediction] += 1
                        elif label == 1:
                            data_matrix[1, prediction] += 1
                        elif label == 2:
                            data_matrix[2, prediction] += 1

            # print accuracy for each class
            total = 0
            print("Epoch:")
            print(epoch)
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                total = total + accuracy
                print("Test Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            print("Average Test Accuracy is: {:.1f} %".format(total / 3))
            test_acc.append(total / 3)

        if epoch % 50 == 0:
            classes = ["phantom1", "phantom2", "phantom3"]
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            data_matrix = np.zeros((3, 3))
            net.eval()

            # again no gradients needed
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels = data
                    # inputs = inputs.float()
                    # labels = labels.long()
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    outputs = net(inputs)
                    _, predictions = torch.max(outputs, 1)
                    # collect the correct predictions for each class
                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
                        if label == 0:
                            data_matrix[0, prediction] += 1
                        elif label == 1:
                            data_matrix[1, prediction] += 1
                        elif label == 2:
                            data_matrix[2, prediction] += 1

            # print accuracy for each class
            total = 0
            print("Epoch:")
            print(epoch)
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                total = total + accuracy
                print("Valid Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            print("Average Valid Accuracy is: {:.1f} %".format(total / 3))
            validation_acc.append(total / 3)

        print("Execution time is %s seconds" % (time.time() - start_time))

    print('Finished Training')
    torch.save(net.state_dict(), PATH)
    return mean_data, std_data, validation_acc, test_acc


def dataset10_28_21_single_settings_3centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_3patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_3patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_3patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_3centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_3centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_4centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_4patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_4patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_4patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_4centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_4centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_5centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_5patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_5patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_5patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_5centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_5centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_6centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_6patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_6patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_6patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_6centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_6centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_7centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_7patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_7patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_7patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_7centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_7centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_8centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_8patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_8patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_8patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_8centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_8centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_9centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_9patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_9patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_9patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_9centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_9centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_10centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_10patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_10patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_10patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_10centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_10centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_11centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_11patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_11patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_11patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_11centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_11centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def dataset10_28_21_single_settings_13centerpatches_for_setsize(coef, num):
    num = num//3

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class1 = far_field_13patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class2 = far_field_13patches(volume, coef * depth)

    filepath1 = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(filepath1)
    depth, channels, frames = volume.shape
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    class3 = far_field_13patches(volume, coef * depth)

    return np.concatenate((class1, class2, class3), axis=0),\
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0])+1, np.zeros(class3.shape[0])+2))


def dataset10_28_21_single_settings_13centerpatches_for_setsize_train_test_split(coef, num):
    x, y = dataset10_28_21_single_settings_13centerpatches_for_setsize(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def RegularTraining(num, Depth, Start):
    num = num // 3

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    class1, depth1 = extract_all_patchess(fileName, num, Depth, Start)
    indices = np.random.permutation(class1.shape[0])
    class1 = class1[indices, :, :]
    depth1 = depth1[indices]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    class2, depth2 = extract_all_patchess(fileName, num, Depth, Start)
    indices = np.random.permutation(class2.shape[0])
    class2 = class2[indices, :, :]
    depth2 = depth2[indices]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    class3, depth3 = extract_all_patchess(fileName, num, Depth, Start)
    indices = np.random.permutation(class3.shape[0])
    class3 = class3[indices, :, :]
    depth3 = depth3[indices]

    return np.concatenate((class1, class2, class3), axis=0), \
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1, np.zeros(class3.shape[0]) + 2)), \
           np.concatenate((depth1, depth2, depth3), axis=0)


def RegularTraining_train_test_split(num, Depth, Start):
    x, y, depth = RegularTraining(num, Depth, Start)
    x_train = x[::3]
    y_train = y[::3]
    depth_train = depth[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    depth_test = depth[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    depth_valid = depth[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid, depth_train, depth_test, depth_valid


def ZoneTraining(coef, num):
    num = num // 3

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(fileName)
    depth, channels, frames = volume.shape
    print(volume.shape)
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]

    class1 = far_field_3patches(volume, coef * depth)
    indices = np.random.permutation(class1.shape[0])
    class1 = class1[indices, :, :]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(fileName)
    depth, channels, frames = volume.shape
    print(volume.shape)
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]

    class2 = far_field_3patches(volume, coef * depth)
    indices = np.random.permutation(class2.shape[0])
    class2 = class2[indices, :, :]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(fileName)
    depth, channels, frames = volume.shape
    print(volume.shape)
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]

    class3 = far_field_3patches(volume, coef * depth)
    indices = np.random.permutation(class3.shape[0])
    class3 = class3[indices, :, :]

    return np.concatenate((class1, class2, class3), axis=0), \
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1, np.zeros(class3.shape[0]) + 2))


def ZoneTraining_train_test_split(coef, num):
    x, y = ZoneTraining(coef, num)
    x_train = x[::3]
    y_train = y[::3]
    x_test = x[1::3]
    y_test = y[1::3]
    x_valid = x[2::3]
    y_valid = y[2::3]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid


if __name__ == '__main__':
    PATH = './saved_models/center_trained_model.pth'
    # us_images = [250, 375, 500, 625, 750, 875]  #training size[200, 300, 400, 500, 600, 700]
    # us_images = [62, 125, 250, 375, 500]  #training size[50, 100, 200, 300, 400]
    # us_images = [125, 250, 375, 500, 625]  # training size[100, 200, 300, 400, 500]

    for number, Start, Depth, epoch in zip(us_images, START_pixel, patch_width, epoch_list):
        print("US image number:")
        print(number)
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        for i in range(repetition):
            print("Trial:")
            print(i)
            print("Training")
            x_train, x_test, x_valid, y_train, y_test, y_valid, depth_train, depth_test, depth_valid = \
                RegularTraining_train_test_split(number, Depth, Start)
            print(x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape)
            mean_data, std_data, validation_acc, test_acc = train_function(x_train, x_valid, x_test, y_train, y_valid,
                                                                           y_test, PATH, epoch, LR)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print(test_acc)
            print(temp)
            c1.append(temp)

            print("Test with pre")
            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split(1.4 / 4, number)
            print(x_test.shape, y_test.shape)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print(temp)
            c2.append(temp)

            print("Test with on")
            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split(2 / 4, number)
            print(x_test.shape, y_test.shape)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print(temp)
            c3.append(temp)

            print("Test with post")
            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split(2.6 / 4, number)
            print(x_test.shape, y_test.shape)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print(temp)
            c4.append(temp)

        print("US image number:")
        print(number)
        print("Training and Testing the Same")
        arr_r = np.array(c1)
        print(np.mean(arr_r, axis=0))
        print(np.std(arr_r, axis=0))
        print("Pre Focal")
        arr_r = np.array(c2)
        print(np.mean(arr_r, axis=0))
        print(np.std(arr_r, axis=0))
        print("On Focal")
        arr_r = np.array(c3)
        print(np.mean(arr_r, axis=0))
        print(np.std(arr_r, axis=0))
        print("Post Focal")
        arr_r = np.array(c4)
        print(np.mean(arr_r, axis=0))
        print(np.std(arr_r, axis=0))

        # for i in range(repetition):
        #     print("Trial:")
        #     print(i)
        #     print("5 patches")
        #     x_train, x_test, x_valid, y_train, y_test, y_valid = dataset10_28_21_single_settings_5centerpatches_for_setsize_train_test_split\
        #         (center_loc/4, number)
        #     print(x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape)
        #     _, _, _, test_acc = train_function(x_train, x_valid, x_test, y_train, y_valid,
        #                                        y_test, PATH, epoch[0], LR)
        #     print(test_acc)
        #     c1.append(test_acc)
        #
        #     print("9 patches")
        #     x_train, x_test, x_valid, y_train, y_test, y_valid = dataset10_28_21_single_settings_9centerpatches_for_setsize_train_test_split\
        #         (center_loc/4, number)
        #     print(x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape)
        #     _, _, _, test_acc = train_function(x_train, x_valid, x_test, y_train, y_valid,
        #                                        y_test, PATH, epoch[1], LR)
        #     print(test_acc)
        #     c2.append(test_acc)
        #
        #     print("13 patches")
        #     x_train, x_test, x_valid, y_train, y_test, y_valid = dataset10_28_21_single_settings_13centerpatches_for_setsize_train_test_split\
        #         (center_loc/4, number)
        #     print(x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape)
        #     _, _, _, test_acc = train_function(x_train, x_valid, x_test, y_train, y_valid,
        #                                        y_test, PATH, epoch[2], LR)
        #     print(test_acc)
        #     c3.append(test_acc)
        #
        # print("US image number:")
        # print(number)
        # print("5 patches")
        # arr_r = np.array(c1)
        # print(np.mean(arr_r, axis=0))
        # print(np.std(arr_r, axis=0))
        # print("9 patches")
        # arr_r = np.array(c2)
        # print(np.mean(arr_r, axis=0))
        # print(np.std(arr_r, axis=0))
        # print("13 patches")
        # arr_r = np.array(c3)
        # print(np.mean(arr_r, axis=0))
        # print(np.std(arr_r, axis=0))


