import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
# from prepare_utils import *
from netmodel import *
from read_rf import read
from denoising import Denoise_us
import time
import statistics

Training_Visual = False
Test_Visual = False
Normalization_FLAG = False  # Patchwise
Normalization_FLAG2 = False  # scanlinewise
Normalization_FLAG3 = False  # Framewise
Noise_FLAG = False
Print_Flag = False
Noise_low = 6
Noise_high = -60
learning_rate = [4e-6]
denoising_func = Denoise_us(low=Noise_low, high=Noise_high)
us_images = [int(300 * (1))]
epochs = [800]
repetition = 4


# Figure 8 in Zone Training Paper Arxiv version
def extract_all_patchess(filepath, num):
    volume = read(filepath)
    if Noise_FLAG:
        volume = denoising_func(volume)

    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    depth, channels, frames = volume.shape

    if channels == 512:
        volume = volume[:, ::2, :]

    if Normalization_FLAG2:
        mean = np.mean(volume[250:1750, :, :], axis=0)
        std = np.std(volume[250:1750, :, :], axis=0)
        volume = (volume - mean[np.newaxis, :, :]) / std[np.newaxis, :, :]

    if Normalization_FLAG3:
        mean = np.mean(volume[250:1750, :, :].reshape(1500 * 256, -1), axis=0)
        std = np.std(volume[250:1750, :, :].reshape(1500 * 256, -1), axis=0)
        volume = (volume - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]

    start_depth = 400
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
        if depth_counter == 12:
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


def test_function(x_test, y_test, depth_test, mean_data, std_data, PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Using {} device".format(device))

    # net = MNet().to(device)
    net = AlexNet(3).to(device)

    # print(net)
    # parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters:{parameter_number}")
    mean_data_ts = np.mean(x_test.reshape(-1, 256 * 256), axis=1)
    std_data_ts = np.std(x_test.reshape(-1, 256 * 256), axis=1)

    net.load_state_dict(torch.load(PATH))

    if Normalization_FLAG:
        x_test = (x_test - mean_data_ts[:, np.newaxis, np.newaxis]) / std_data_ts[:, np.newaxis, np.newaxis]

    dataset = TensorDataset(torch.from_numpy(x_test[:, np.newaxis, :, :]), torch.from_numpy(y_test),
                            torch.from_numpy(depth_test))
    test_loader = DataLoader(dataset, batch_size=64, pin_memory=True, shuffle=True)

    # prepare to count predictions for each class
    classes = ["phantom1", "phantom2", "phantom3"]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    data_matrix = np.zeros((3, 3))
    depth_matrix = np.zeros((14, 3))
    net.eval()

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, depth = data
            inputs = inputs.float()
            labels = labels.long()
            depth = depth.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            depth = depth.to(device)
            outputs = net(inputs)
            depth = depth.cpu().detach().numpy()
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction, depth_index in zip(labels, predictions, depth):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    depth_matrix[int(depth_index), 0] = depth_matrix[int(depth_index), 0] + 1
                else:
                    depth_matrix[int(depth_index), 1] = depth_matrix[int(depth_index), 1] + 1
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

    print("Average Accuracy is: {:.1f} %".format(total / 3))
    if Test_Visual:
        print("Confusion Matrix:")
        print(data_matrix)
        plt.matshow(data_matrix)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.show()

        depth_matrix[:, 2] = depth_matrix[:, 0] / (depth_matrix[:, 0] + depth_matrix[:, 1])
        # depth_list = [1.02, 1.21, 1.4, 1.59, 1.78, 1.98, 2.17, 2.36, 2.55, 2.75, 2.94, 3.13]
        depth_list = [1.02, 1.21, 1.4, 1.59, 1.78, 1.98, 2.17, 2.36, 2.55, 2.75, 2.94, 3.13, 3.32, 3.52]

        print("Depth Matrix:")
        print(depth_matrix)
        # plt.plot(depth_list, depth_matrix[:14, 2])
        plt.plot(depth_list, depth_matrix[:, 2])
        plt.title("Accuracy vs Patch Depth")
        plt.xlabel("Depth in cm")
        plt.ylabel("Classification Accuracy")
        plt.grid()
        plt.show()
    return total / 3


def train_function(x_train, x_valid, x_test, y_train, y_valid, y_test, depth_train, depth_valid, depth_test, PATH,
                   epoch_num, LR):
    accuracies = []
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Calculate Mean
    # mean_data = np.mean(x_train.reshape(-1, 1), axis=0)
    # std_data = np.std(x_train.reshape(-1, 1), axis=0)

    # # Calculate Mean
    mean_data = np.mean(x_train, axis=0)
    std_data = np.std(x_train, axis=0)

    x_valid = (x_valid - mean_data) / std_data
    x_train = (x_train - mean_data) / std_data
    x_test = (x_test - mean_data) / std_data

    # mean_data = np.mean(x_train.reshape(-1, 256*256), axis=1)
    # std_data = np.std(x_train.reshape(-1, 256*256), axis=1)
    # mean_data_ts = np.mean(x_test.reshape(-1, 256*256), axis=1)
    # std_data_ts = np.std(x_test.reshape(-1, 256*256), axis=1)
    #
    # # z-score normalization or standardization
    # if Normalization_FLAG:
    #     x_train = (x_train - mean_data[:, np.newaxis, np.newaxis]) / std_data[:, np.newaxis, np.newaxis]
    #     x_test = (x_test - mean_data_ts[:, np.newaxis, np.newaxis]) / std_data_ts[:, np.newaxis, np.newaxis]
    x_train_gpu = torch.from_numpy(x_train[:, np.newaxis, :, :]).float().to("cuda")
    y_train_gpu = torch.from_numpy(y_train).long().to("cuda")
    x_valid_gpu = torch.from_numpy(x_valid[:, np.newaxis, :, :]).float().to("cuda")
    y_valid_gpu = torch.from_numpy(y_valid).long().to("cuda")
    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to("cuda")
    y_test_gpu = torch.from_numpy(y_test).long().to("cuda")

    dataset = TensorDataset(x_train_gpu, y_train_gpu)
    train_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    dataset = TensorDataset(x_valid_gpu, y_valid_gpu)
    valid_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    dataset = TensorDataset(x_test_gpu, y_test_gpu)
    test_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    net = AlexNet_review(3).to(device)
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
    if Training_Visual:
        plt.plot(loss_epoch)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()

        plt.plot(validation_acc)
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()
    torch.save(net.state_dict(), PATH)
    return mean_data, std_data, validation_acc, test_acc


def RegularTraining(num):
    num = num // 3

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    class1, depth1 = extract_all_patchess(fileName, num)
    indices = np.random.permutation(class1.shape[0])
    class1 = class1[indices, :, :]
    depth1 = depth1[indices]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    class2, depth2 = extract_all_patchess(fileName, num)
    indices = np.random.permutation(class2.shape[0])
    class2 = class2[indices, :, :]
    depth2 = depth2[indices]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    class3, depth3 = extract_all_patchess(fileName, num)
    indices = np.random.permutation(class3.shape[0])
    class3 = class3[indices, :, :]
    depth3 = depth3[indices]

    return np.concatenate((class1, class2, class3), axis=0), \
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1, np.zeros(class3.shape[0]) + 2)), \
           np.concatenate((depth1, depth2, depth3), axis=0)


def RegularTraining_train_test_split(num):
    x, y, depth = RegularTraining(num)
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


if __name__ == '__main__':
    PATH = './saved_models/regular.pth'

    for number, epoch, LR in zip(us_images, epochs, learning_rate):
        print("US image number:")
        print(number)
        store = []
        r = []

        for i in range(repetition):
            print("Trial:")
            print(i + 1)
            x_train, x_test, x_valid, y_train, y_test, y_valid, depth_train, depth_test, depth_valid = RegularTraining_train_test_split(
                number)
            print(f'Training Set Size is {x_train.shape[0]} in terms of US')
            mean_data, std_data, valid_acc, test_acc = train_function(x_train, x_valid, x_test, y_train, y_valid,
                                                                      y_test, depth_train, depth_valid, depth_test,
                                                                      PATH, epoch, LR)
            r.append(test_acc)
            # temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            # r.append(temp)

        print("US image number:")
        print(number)
        print("Results:")
        arr_r = np.array(r)
        print(np.mean(arr_r, axis=0))
        print(np.std(arr_r, axis=0))

