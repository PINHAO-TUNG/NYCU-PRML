import numpy as np
from datetime import datetime
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.metrics import accuracy_score


class FromNPY(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = torch.from_numpy(target).view(-1).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # Hyper-parameters
    batch_size = 64
    lr = 1e-2  # 1e-4
    weight_decay = 5e-4
    num_classes = 10
    total_epoch = 100
    print_per_iteration = 100

    # Data Load
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # It's a multi-class classification problem
    class_index = {
        'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(np.unique(y_train))
    print(device)

    
    # ##Model
    # transformer
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ColorJitter(0.1, 0.3, 0.3, 0.3),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]
    )
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]
    )

    train_dataset = FromNPY(x_train, y_train, transform=train_transforms)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = FromNPY(x_test, y_test, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    PATH = './310581020_HW5_model.pth'
    model = torch.load(PATH)
    model.eval()
    correct = 0
    total = 0
    y_pred = None
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            if y_pred is None:
                y_pred = np.array(predicted.cpu().numpy())
            else:
                y_pred = np.append(y_pred, predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

    # ## DO NOT MODIFY CODE BELOW!
    # please screen shot your results and post it on your report
    assert y_pred.shape == (10000,)
    print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))