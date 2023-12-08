import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from torchinfo import summary
from torch.utils.data import DataLoader

def gpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def load_images(path):
    images = []
    filenames = os.listdir(path)
    print(filenames)
    for filename in tqdm(filenames):
        if filename == '_DS_Store':
            continue
        image = cv2.imread(os.path.join(path,filename))
        image = cv2.resize(image, dsize=(100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    # print(images)
    return np.array(images)

def image_path():
    cats_train_images = load_images('animal_dataset/training_set/training_set/cats')
    dogs_train_images = load_images('animal_dataset/training_set/training_set/dogs')
    cats_test_images = load_images('animal_dataset/test_set/test_set/cats')
    dogs_test_images = load_images('animal_dataset/test_set/test_set/dogs')

    # cats_train의 샘플 수와 동일한 길이의 0으로 이루어진 리스트를 생성합니다.
    # len(dogs_train): 이 부분은 dogs_train 배열의 샘플 수를 계산
    # [1] * len(dogs_train): 이것은 dogs_train의 샘플 수와 동일한 길이의 1로 이루어진 리스트를 생성
    # : 두 리스트(0으로 이루어진 것과 1로 이루어진 것)는 + 연산자를 사용하여 연결    # 배열의 첫 번째 절반은 cats_train 샘플에 해당하며 0으로 레이블이 지정되고, 두 번째 절반은 dogs_train 샘플에 해당하며 1로 레이블이 지정
    X_train = np.append(cats_train_images, dogs_train_images, axis=0)
    X_test = np.append(cats_test_images, dogs_test_images, axis=0)

    y_train = np.array([0] * len(cats_train_images) + [1] * len(dogs_train_images))
    y_test = np.array([0] * len(cats_test_images) + [1] * len(dogs_test_images))

    return cats_train_images, dogs_train_images, cats_test_images, dogs_test_images, X_train, X_test, y_train, y_test


def show_images(images, labels, start_index):
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20,12))

    counter = start_index

    for i in range(4):
        for j in range(8):
            axes[i,j].set_title(labels[counter].item())
            axes[i,j].imshow(images[counter], cmap='gray')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
            counter += 1
    plt.show()

def preprocessing():
    y_train, y_test = image_path()[-2], image_path()[-1]
    y_train = torch.from_numpy(y_train.reshape(len(y_train), 1))
    y_test = torch.from_numpy(y_test.reshape(len(y_test), 1))
    # print(y_train[:10]) # Checking normalizing result.

    transforms_train = transforms.Compose([transforms.ToTensor(),  # convert to Tensor.
                                           transforms.RandomRotation(degrees=20),  # random rotation
                                           # p parameter represents the probability that the transformation function is applied to the image.
                                           transforms.RandomHorizontalFlip(p=0.5),  # random horizontal flip
                                           transforms.RandomVerticalFlip(p=0.005),  # random vertical flip
                                           transforms.RandomGrayscale(p=0.2),  # random grayscale
                                           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # squeeze to -1 and 1
                                           ])

    transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    return transforms_train, transforms_test


class CustomData:
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self): # 데이터세트 내의 샘플 수를 알아내기 위해
        return len(self.images)

    def __getitem__(self, index): # 클래스의 객체를 색인화 가능하게 만든다.
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return (image, label)

    # 32개 미만의 이미지가 포함될 때마다 마지막 배치가 모델에 공급되는 것을 방지하기 위해 drop_last 매개변수를 True로 설정

def dataloader():
    transforms_train, transforms_test = preprocessing()
    X_train, y_train, X_test, y_test = image_path()[-4], image_path()[-2], image_path()[-3], image_path()[-1]

    train_dataset = CustomData(images=X_train, labels=y_train, transform=transforms_train)
    test_dataset = CustomData(images=X_test, labels=y_test, transform=transforms_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    iterator = iter(train_loader)
    image_batch, label_batch = next(iterator)
    # print(image_batch.shape)

    image_batch_permuted = image_batch.permute(0, 2, 3, 1)
    print(image_batch_permuted.shape)
    show_images(image_batch_permuted, label_batch, 0)
    return train_loader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        # self.maxpool

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # self.maxpool

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        # self.maxpool

        self.dropout = nn.Dropout(p=0.5)
        self.fc0 = nn.Linear(in_features=128 * 6 * 6, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x

def model_training(model, train_loader, optimizer, loss_function):
    num_correct_train = 0
    num_samples_train = 0

    for batch, (X_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        device = gpu()
        X_train = X_train.float().to(device)
        y_train = y_train.float().to(device)

        # Forward propagation
        train_preds = model(X_train)
        train_loss = loss_function(train_preds, y_train)

        # Calculate train accuracy
        with torch.no_grad():
            rounded_train_preds = torch.round(train_preds)
            num_correct_train += torch.sum(rounded_train_preds == y_train)
            num_samples_train += len(y_train)

        # Backward propagation
        optimizer.zero_grad()
        train_loss.backward()

        # Gradient descent
        optimizer.step()

    train_acc = num_correct_train / num_samples_train
    return train_loss, train_acc

    # 테스트 데이터를 예측하는 기능

def predict_test_data(model, test_loader):
    num_correct = 0
    num_sample = 0
    model.evel() # model 평가 모드
    # 네트워크가 최상의 성능을 얻을 수 있도록 드롭아웃 계층에서 무작위로 연결이 끊긴 뉴런을 다시 연결하기 위해서

    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.float().to(device)
            y_test = y_test.float().to(device)

            # Calculate loss (forward propagation)
            test_preds = model(X_test)
            test_loss = loss_function(test_preds, y_test)

            # Calculate accuracy
            rounded_test_preds = torch.round(test_preds)
            num_correct += torch.sum(rounded_test_preds == y_test)
            num_samples += len(y_test)

        model.train()

        test_acc = num_correct / num_samples
        return test_loss, test_acc

def start_training(model):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    transforms_train, transforms_test = preprocessing()
    X_train, y_train, X_test, y_test = image_path()[-4], image_path()[-2], image_path()[-3], image_path()[-1]

    train_dataset = CustomData(images=X_train, labels=y_train, transform=transforms_train)
    test_dataset = CustomData(images=X_test, labels=y_test, transform=transforms_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCELoss()

    for epoch in range(100):
        train_loss, train_acc = model_training(model, train_loader, optimizer, loss_function)
        test_loss, test_acc = predict_test_data(model, test_loader)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())

        print(f'Epoch: {epoch} \t|' \
              f' Train loss: {np.round(train_loss.item(), 3)} \t|'
              f' Test loss: {np.round(test_loss.item(), 3)} \t|'
              f' Train acc: {np.round(train_acc.item(), 2)} \t'
              f' Test acc: {np.round(test_acc.item(), 2)}')

if __name__ == '__main__':
    dataloader()
    model = CNN().to(gpu()) # cal for using GPU.
    # summary(model, input_size=(4, 3, 100, 100)) # summary: 모델 세부정보 인쇄
    start_training(model)
# result = image_path()
# if result is not None:
    # X_train, y_train = result[-3], result[-1]
    # show_images(X_train, y_train, 0)
    # print(y_train[:10])


# 건너뛴 _DS_Store 파일 덕분에 실제 이미지 수를 얻으려면 1을 빼야 한다.