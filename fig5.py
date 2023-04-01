import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
from prepare_data import *
from netmodel import *
from median_pool import *
import statistics
import time

#Large:(1e-4,200,97.4) (5e-5,150,97.3) (1e-5,700,96.2), (5e-6,800,96.1), (1e-6,800,94.6)
learning_rate = 1e-5
us_images = [int(500*(3))]
epochs = [400]
repetition = 5
center_loc = 2.6


# Fig5 and Fig4 in Zone Training Paper Arxiv version
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
                            volume[focus - 100 - 100: focus - 100 + 100, 114:140, :],
                            volume[focus - 100: focus + 100, 114:140, :],
                            volume[focus + 100 - 100: focus + 100 + 100, 114:140, :],
                            volume[focus - 100 - 100: focus - 100 + 100, 140:166, :],
                            volume[focus - 100: focus + 100, 140:166, :],
                            volume[focus + 100 - 100: focus + 100 + 100, 140:166, :],
                            volume[focus - 100 - 100: focus - 100 + 100, 166:192, :],
                            volume[focus - 100: focus + 100, 166:192, :],
                            volume[focus + 100 - 100: focus + 100 + 100, 166:192, :],
                            volume[focus - 100 - 100: focus - 100 + 100, 192:218,:],
                            volume[focus - 100: focus + 100, 192:218,:],
                            volume[focus + 100 - 100: focus + 100 + 100, 192:218,:],
                            volume[focus - 100 - 100: focus - 100 + 100, 218:244, :],
                            volume[focus - 100: focus + 100, 218:244, :],
                            volume[focus + 100 - 100: focus + 100 + 100, 218:244, :]
                              ), axis=2)

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
    test_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

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


def train_function(x_train, x_test, y_train, y_test, PATH, epoch_num):
    accuracies = []
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Calculate Mean
    mean_data = np.mean(x_train, axis=0)
    std_data = np.std(x_train, axis=0)

    # z-score normalization or standardization
    x_train = (x_train-mean_data)/std_data
    x_test = (x_test-mean_data)/std_data

    x_train_gpu = torch.from_numpy(x_train[:, np.newaxis, :, :]).float().to("cuda")
    y_train_gpu = torch.from_numpy(y_train).long().to("cuda")
    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to("cuda")
    y_test_gpu = torch.from_numpy(y_test).long().to("cuda")

    dataset = TensorDataset(x_train_gpu, y_train_gpu)
    train_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    dataset = TensorDataset(x_test_gpu, y_test_gpu)
    test_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    # for X, y in train_loader:
    #     print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # net = MNet().to(device)
    net = AlexNet_review(3).to(device)
    # print(net)
    parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters:{parameter_number}")

    criterion = nn.CrossEntropyLoss(torch.tensor([1, 1, 1]).float().to(device))
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
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

        print("Execution time is %s seconds" % (time.time() - start_time))

    print('Finished Training')

    torch.save(net.state_dict(), PATH)
    # # prepare to count predictions for each class
    # classes = ["phantom1", "phantom2", "phantom3"]
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}
    # data_matrix = np.zeros((3, 3))
    # net.eval()
    #
    # # again no gradients needed
    # with torch.no_grad():
    #     for data in train_loader:
    #         inputs, labels = data
    #         inputs = inputs.float()
    #         labels = labels.long()
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         # inputs = med_filter(inputs)
    #
    #         outputs = net(inputs)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1
    #             if label == 0:
    #                 data_matrix[0, prediction] += 1
    #             elif label == 1:
    #                 data_matrix[1, prediction] += 1
    #             elif label == 2:
    #                 data_matrix[2, prediction] += 1

    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

    return mean_data, std_data, accuracies


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

    for number, epoch in zip(us_images, epochs):
        print("US image number:")
        print(number)
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        c5 = []
        c6 = []
        c7 = []
        c8 = []
        c9 = []

        for i in range(repetition):
            print("Trial:")
            print(i)
            print("Center")
            x_train, x_test, x_valid, y_train, y_test, y_valid = ZoneTraining_train_test_split(center_loc/4, number)
            print(x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape)
            mean_data, std_data, acc = train_function(x_train, x_valid, y_train, y_valid, PATH, epoch)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("0")
            print(temp)
            c1.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc+0.2)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("0.25")
            print(temp)
            c2.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc+0.4)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("0.5")
            print(temp)
            c3.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc+0.6)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("0.75")
            print(temp)
            c4.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc+0.8)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("1")
            print(temp)
            c5.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc-0.2)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("-0.25")
            print(temp)
            c6.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc-0.4)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("-0.5")
            print(temp)
            c7.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc-0.6)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("-0.75")
            print(temp)
            c8.append(temp)

            _, x_test, _, _, y_test, _ = ZoneTraining_train_test_split((center_loc-0.8)/4, number)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print("-1")
            print(temp)
            c9.append(temp)

        print("US image number:")
        print(number)
        print("0")
        print(sum(c1)/len(c1))
        print(statistics.pstdev(c1))
        print("0.2")
        print(sum(c2)/len(c2))
        print(statistics.pstdev(c2))
        print("0.4")
        print(sum(c3)/len(c3))
        print(statistics.pstdev(c3))
        print("0.6")
        print(sum(c4)/len(c4))
        print(statistics.pstdev(c4))
        print("0.8")
        print(sum(c5)/len(c5))
        print(statistics.pstdev(c5))
        print("-0.2")
        print(sum(c6)/len(c6))
        print(statistics.pstdev(c6))
        print("-0.4")
        print(sum(c7)/len(c7))
        print(statistics.pstdev(c7))
        print("-0.6")
        print(sum(c8)/len(c8))
        print(statistics.pstdev(c8))
        print("-0.8")
        print(sum(c9)/len(c9))
        print(statistics.pstdev(c9))
