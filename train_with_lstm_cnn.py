import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


class GestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.label_list = []

        environments = os.listdir(root_dir)
        for env in environments:
            env_path = os.path.join(root_dir, env)
            gestures = os.listdir(env_path)
            for gesture_idx, gesture in enumerate(gestures):
                gesture_path = os.path.join(env_path, gesture)
                for img_name in os.listdir(gesture_path):
                    self.file_list.append(os.path.join(gesture_path, img_name))
                    self.label_list.append(gesture_idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class LSTMCNN(nn.Module):
    def __init__(self, num_gestures=7):
        super(LSTMCNN, self).__init__()
        self.lstm = nn.LSTM(3, 64, num_layers=3, batch_first=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.bn8 = nn.BatchNorm1d(1000)
        self.fc4 = nn.Linear(1000, num_gestures)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # LSTM层
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(-1))
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), 224, 224, -1)
        x = x.permute(0, 3, 1, 2)

        # 卷积
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        # FC
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.bn6(self.fc1(x))))
        x = self.dropout(self.relu(self.bn7(self.fc2(x))))
        x = self.dropout(self.relu(self.bn8(self.fc3(x))))
        x = self.fc4(x)

        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

    return train_losses, train_accuracies, val_losses, val_accuracies


def main():
    train_dir = 'datasets/test'
    test_dir = 'datasets/train'
    batch_size = 32
    num_epochs = 50
    num_gestures = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GestureDataset(train_dir, transform=transform)
    test_dataset = GestureDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = LSTMCNN(num_gestures=num_gestures).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device
    )

    torch.save(model.state_dict(), 'lstm_cnn_gesture_model.pth')

    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_gesture.png')
    plt.show()


if __name__ == '__main__':
    main()