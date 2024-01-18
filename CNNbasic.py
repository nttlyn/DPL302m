from google.colab import drive
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset  
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
import numpy as np

drive.mount('/content/drive')
device=torch.device("cuda"if torch.cuda.ii_available()else "cpu")
image_folder = "/content/drive/MyDrive/hunglabel/image"
label_folder = "/content/drive/MyDrive/hunglabel/labels"


image_files = os.listdir(image_folder)
label_files = os.listdir(label_folder)

image_names = [os.path.splitext(file)[0] for file in image_files]

train_names, test_names = train_test_split(image_names, test_size=0.25, random_state=42)

print("Số lượng ảnh trong train:", len(train_names))
print("Số lượng ảnh trong val:", len(test_names))
def load_data(image_folder, label_folder, image_names, max_label_length):
    images = []
    labels = []

    for name in image_names:
        image_path = os.path.join(image_folder, name + ".png")
        img = cv2.imread(image_path)
        images.append(img)

        label_path = os.path.join(label_folder, name + ".txt")
        with open(label_path, "r") as file:
            lines = file.readlines()
            label = [list(map(float, line.strip().split())) for line in lines]
            label = label + [[0.0] * len(label[0])] * (max_label_length - len(label))
            labels.append(label)

    return np.array(images), np.array(labels)
max_label_length = max(len(label) for label in labels)
images, labels = load_data(image_folder, label_folder, train_names, max_label_length)
images = np.array(images) / 255.0
images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device) 
labels = [torch.tensor(label, dtype=torch.float32).to(device) for label in labels]
labels = torch.stack(labels)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        torch.cuda.empty_cache()
        return x

model = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train_dataset = TensorDataset(images, labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

epochs = 10
for epoch in range(epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

