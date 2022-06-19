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

    # CNN Model
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, num_classes)
    model.convl = nn.Conv2d(3, 64, kernel_size=(
        3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.classifier = nn.Linear(
        in_features=1024, out_features=10, bias=False)
    model.to(device)
    # print(model)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr,
    # weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    save_path = './310581020_HW5_model.pth'

    model.eval()
    print(datetime.now())
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        if epoch <= 90:
            lr = 0.000001 * epoch * epoch - 0.0002 * epoch + 0.01
            set_learning_rate(optimizer, lr)
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if (i+1) % print_per_iteration == 0:    # print every 2000 mini-batches
                print(
                    f'[ep {epoch + 1}][{i + 1:5d}/{len(train_loader):5d}] loss: {loss.item():.3f}')
        # scheduler.step()
        torch.save(model, save_path)
        print(datetime.now())

    # Evaluate model
    # fixed testing process
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

    torch.save(model.state_dict(), "./310581020_HW5_model.pt")
    # ## DO NOT MODIFY CODE BELOW!
    # please screen shot your results and post it on your report
    assert y_pred.shape == (10000,)
    print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))
