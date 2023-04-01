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
from sklearn.model_selection import train_test_split
import statistics
import time


Training_Visual = False
us_images = [int(25 * (3))]
epochs = [2000]
learning_rates = [5e-6]
repetition = 10
Shift = 40
Depth = 9
Start = 540
center_loc = 1.2


# Table 4,5,6 in Zone Training Paper Arxiv version
def far_field_3patches(volume, focus):
    focus = int(focus)
    no_frames = volume.shape[2]
    depth_ind = focus//100 - 6
    focus = (focus - (focus % 100)) + Shift
    patches = np.concatenate(
        (volume[focus - 100 - 100: focus - 100 + 100, 10:36, :], volume[focus - 100: focus + 100, 10:36, :],
         volume[focus + 100 - 100: focus + 100 + 100, 10:36, :],
         volume[focus - 100 - 100: focus - 100 + 100, 36:62, :], volume[focus - 100: focus + 100, 36:62, :],
         volume[focus + 100 - 100: focus + 100 + 100, 36:62, :],
         volume[focus - 100 - 100: focus - 100 + 100, 62:88, :], volume[focus - 100: focus + 100, 62:88, :],
         volume[focus + 100 - 100: focus + 100 + 100, 62:88, :],
         volume[focus - 100 - 100: focus - 100 + 100, 88:114, :], volume[focus - 100: focus + 100, 88:114, :],
         volume[focus + 100 - 100: focus + 100 + 100, 88:114, :],
         volume[focus - 100 - 100: focus - 100 + 100, 114:140, :], volume[focus - 100: focus + 100, 114:140, :],
         volume[focus + 100 - 100: focus + 100 + 100, 114:140, :],
         volume[focus - 100 - 100: focus - 100 + 100, 140:166, :], volume[focus - 100: focus + 100, 140:166, :],
         volume[focus + 100 - 100: focus + 100 + 100, 140:166, :],
         volume[focus - 100 - 100: focus - 100 + 100, 166:192, :], volume[focus - 100: focus + 100, 166:192, :],
         volume[focus + 100 - 100: focus + 100 + 100, 166:192, :],
         volume[focus - 100 - 100: focus - 100 + 100, 192:218, :], volume[focus - 100: focus + 100, 192:218, :],
         volume[focus + 100 - 100: focus + 100 + 100, 192:218, :],
         volume[focus - 100 - 100: focus - 100 + 100, 218:244, :], volume[focus - 100: focus + 100, 218:244, :],
         volume[focus + 100 - 100: focus + 100 + 100, 218:244, :]), axis=2)

    depth_patch = np.concatenate((np.full((200, 26, no_frames), (depth_ind-1)/(Depth-1)),
                                  np.full((200, 26, no_frames), depth_ind/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind+1)/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind-1)/(Depth-1)),
                                  np.full((200, 26, no_frames), depth_ind/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind+1)/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind-1)/(Depth-1)),
                                  np.full((200, 26, no_frames), depth_ind/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind+1)/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind-1)/(Depth-1)),
                                  np.full((200, 26, no_frames), depth_ind/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind+1)/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind-1)/(Depth-1)),
                                  np.full((200, 26, no_frames), depth_ind/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind+1)/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind-1)/(Depth-1)),
                                  np.full((200, 26, no_frames), depth_ind/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind+1)/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind-1)/(Depth-1)),
                                  np.full((200, 26, no_frames), depth_ind/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind+1)/(Depth-1)),
                                  np.full((200, 26, no_frames), (depth_ind - 1) / (Depth - 1)),
                                  np.full((200, 26, no_frames), depth_ind / (Depth - 1)),
                                  np.full((200, 26, no_frames), (depth_ind + 1) / (Depth - 1)),
                                  np.full((200, 26, no_frames), (depth_ind - 1) / (Depth - 1)),
                                  np.full((200, 26, no_frames), depth_ind / (Depth - 1)),
                                  np.full((200, 26, no_frames), (depth_ind + 1) / (Depth - 1))), axis=2)

    patches = np.swapaxes(patches, 1, 2)
    patches = np.swapaxes(patches, 0, 1)
    depth_patch = np.swapaxes(depth_patch, 1, 2)
    depth_patch = np.swapaxes(depth_patch, 0, 1)
    return patches, depth_patch


def extract_all_patchess(filepath, num):
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
            depth_list.append(np.full((patch_size, 26), depth_counter/(Depth-1)))

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


def test_function(x_test, y_test, mean_data, std_data, PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Using {} device".format(device))

    net = AlexNet_review(3).to(device)
    # print(net)
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

    print("Average Accuracy is: {:.1f} %".format(total / 3))
    return total / 3


def train_function(x_train, x_valid, x_test, y_train, y_valid, y_test, PATH, epoch_num, LR):
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


def test_function_depth(x_test, y_test, depth_test, mean_data, std_data, PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Using {} device".format(device))

    net = AlexNet_review_depth(3).to(device)
    # print(net)
    # parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters:{parameter_number}")

    net.load_state_dict(torch.load(PATH))

    x_test = (x_test - mean_data) / std_data

    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to("cuda")
    y_test_gpu = torch.from_numpy(y_test).long().to("cuda")
    depth_test_gpu = torch.from_numpy(depth_test[:, np.newaxis, :, :]).float().to("cuda")

    dataset = TensorDataset(x_test_gpu, y_test_gpu, depth_test_gpu)
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
            inputs, labels, depth = data
            outputs = net(inputs, depth)
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

    print("Average Accuracy is: {:.1f} %".format(total / 3))
    return total / 3


def train_function_depth(x_train, x_valid, x_test, y_train, y_valid, y_test, depth_train, depth_valid, depth_test, PATH, epoch_num, LR):
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
    depth_train_gpu = torch.from_numpy(depth_train[:, np.newaxis, :, :]).float().to("cuda")
    x_valid_gpu = torch.from_numpy(x_valid[:, np.newaxis, :, :]).float().to("cuda")
    y_valid_gpu = torch.from_numpy(y_valid).long().to("cuda")
    depth_valid_gpu = torch.from_numpy(depth_valid[:, np.newaxis, :, :]).float().to("cuda")
    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to("cuda")
    y_test_gpu = torch.from_numpy(y_test).long().to("cuda")
    depth_test_gpu = torch.from_numpy(depth_test[:, np.newaxis, :, :]).float().to("cuda")

    dataset = TensorDataset(x_train_gpu, y_train_gpu, depth_train_gpu)
    train_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    dataset = TensorDataset(x_valid_gpu, y_valid_gpu, depth_valid_gpu)
    valid_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    dataset = TensorDataset(x_test_gpu, y_test_gpu, depth_test_gpu)
    test_loader = DataLoader(dataset, batch_size=64 * 2, pin_memory=False, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    net = AlexNet_review_depth(3).to(device)
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
            inputs, labels, depth = data
            # inputs = inputs.float()
            # labels = labels.long()
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Data Augmentation  https://pytorch.org/vision/stable/transforms.html
            inputs = hflipper(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                outputs = net(inputs, depth)
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
                    inputs, labels, depth = data
                    # inputs = inputs.float()
                    # labels = labels.long()
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    outputs = net(inputs, depth)
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
                    inputs, labels, depth = data
                    # inputs = inputs.float()
                    # labels = labels.long()
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    outputs = net(inputs, depth)
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
    depth1 = depth1[indices, :, :]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    class2, depth2 = extract_all_patchess(fileName, num)
    indices = np.random.permutation(class2.shape[0])
    class2 = class2[indices, :, :]
    depth2 = depth2[indices, :, :]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    class3, depth3 = extract_all_patchess(fileName, num)
    indices = np.random.permutation(class3.shape[0])
    class3 = class3[indices, :, :]
    depth3 = depth3[indices, :, :]

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


def ZoneTraining(coef, num):
    num = num // 3

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/phafr9fo2ir30o5d4.rf"
    volume = read(fileName)
    depth, channels, frames = volume.shape
    print(volume.shape)
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]

    class1, depth1 = far_field_3patches(volume, coef * depth)
    indices = np.random.permutation(class1.shape[0])
    class1 = class1[indices, :, :]
    depth1 = depth1[indices, :, :]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph40fr9fo2ir30o5d4.rf"
    volume = read(fileName)
    depth, channels, frames = volume.shape
    print(volume.shape)
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]

    class2, depth2 = far_field_3patches(volume, coef * depth)
    indices = np.random.permutation(class2.shape[0])
    class2 = class2[indices, :, :]
    depth2 = depth2[indices, :, :]

    fileName = "//bi-fs4.beckman.illinois.edu/oelze/home/usoylu2/dataset/o10_28_21/ph50fr9fo2ir30o5d4.rf"
    volume = read(fileName)
    depth, channels, frames = volume.shape
    print(volume.shape)
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]

    class3, depth3 = far_field_3patches(volume, coef * depth)
    indices = np.random.permutation(class3.shape[0])
    class3 = class3[indices, :, :]
    depth3 = depth3[indices, :, :]
    return np.concatenate((class1, class2, class3), axis=0), \
           np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1, np.zeros(class3.shape[0]) + 2)),\
           np.concatenate((depth1, depth2, depth3), axis=0)


def ZoneTraining_train_test_split(coef, num):
    x, y, depth = ZoneTraining(coef, num)
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
    PATH = './saved_models/center_trained_model.pth'

    for number, epoch, LR in zip(us_images, epochs, learning_rates):
        print("US image number:")
        print(number)
        c1 = []
        c2 = []
        c3 = []
        c4 = []

        for i in range(repetition):
            print("Trial:")
            print(i)
            print("Train")
            x_train, x_test, x_valid, y_train, y_test, y_valid, _, _, _ = ZoneTraining_train_test_split(center_loc / 4, number)
            print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
            mean_data, std_data, validation_acc, test_acc = train_function(x_train, x_valid, x_test, y_train, y_valid,
                                                                           y_test, PATH, epoch, LR)
            temp = test_function(x_test, y_test, mean_data, std_data, PATH)
            print(test_acc)
            print(temp)
            c1.append(temp)

        print("US image number:")
        print(number)
        print("Results")
        print(sum(c1) / len(c1))
        print(statistics.pstdev(c1))



