from google.colab import drive

# Kết nối với Google Drive
drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'image')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.image_list = os.listdir(self.image_dir)

        if train:
            self.image_list, _ = train_test_split(self.image_list, test_size=0.2, random_state=42)  

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_list[idx])
        label_name = os.path.join(self.labels_dir, self.image_list[idx].replace('.png', '.txt'))

        image = Image.open(img_name).convert('RGB')
        labels = self.read_labels(label_name)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        image = transform(image)

        return image.to(device), labels.to(device)

    def read_labels(self, label_file):
        with open(label_file, 'r') as file:
            lines = file.readlines()

        labels = []
        num_classes = 1
        max_values = num_classes + 4

        for line in lines:
            data = line.strip().split()
            label = [float(val) for val in data]

            label.extend([0.0] * (max_values - len(label)))

            labels.append(label)

        labels = torch.tensor(labels)
        labels_padded = F.pad(labels, (0, max_values - labels.size(1)))

        return labels_padded.to(device)


class SimpleObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleObjectDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 64 * 64, num_classes + 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            labels = [label.cpu().detach().numpy() for label in labels]

            labels = torch.tensor(np.concatenate(labels, axis=0)).to(device)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
root_dir = '/content/drive/MyDrive/Untitled folder/hunglabel'


image_list = [img.replace('.png', '') for img in os.listdir(os.path.join(root_dir, 'image'))]
train_image_list, test_image_list = train_test_split(image_list, test_size=0.2, random_state=42)

train_dataset = CustomDataset(root_dir=root_dir, transform=transforms.ToTensor(), train=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CustomDataset(root_dir=root_dir, transform=transforms.ToTensor(), train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = SimpleObjectDetectionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=2e-5)
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
