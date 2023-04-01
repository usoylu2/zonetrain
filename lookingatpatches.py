import matplotlib.pyplot as plt
import numpy as np
from read_rf import read
from scipy.signal import hilbert


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


if __name__ == '__main__':

    x_train, x_test, x_valid, y_train, y_test, y_valid = dataset10_28_21_single_settings_10centerpatches_for_setsize_train_test_split(2/4, 100)
    print(x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape)

    log = 20 * np.log10(np.abs(hilbert(x_train[10, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[10])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

    log = 20 * np.log10(np.abs(hilbert(x_train[30, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[30])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

    log = 20 * np.log10(np.abs(hilbert(x_train[40, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[40])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

    log = 20 * np.log10(np.abs(hilbert(x_train[100, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[100])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

    log = 20 * np.log10(np.abs(hilbert(x_train[900, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[1050])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

    log = 20 * np.log10(np.abs(hilbert(x_train[1050, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[1250])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

    log = 20 * np.log10(np.abs(hilbert(x_train[2900, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[2900])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

    log = 20 * np.log10(np.abs(hilbert(x_train[2950, :, :], axis=0)))
    lognorm = log - np.amax(log)
    print(y_train[2950])
    plt.imshow(lognorm, extent=[0, 4, 4, 0], aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.colorbar()
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()