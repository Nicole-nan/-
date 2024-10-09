import os
import torch
from PIL import Image
from torchvision import transforms, models, datasets
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
data_transforms = transforms.Compose([
  transforms.Lambda(lambda x: x.convert('RGB')),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_data = datasets.ImageFolder(root='wheat-leaf-disease', transform=data_transforms)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0
  progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
  for inputs, labels in progress_bar:
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    avg_loss = running_loss / (total / train_loader.batch_size)
    accuracy = 100 * correct / total
    progress_bar.set_postfix({'Loss': avg_loss, 'Accuracy': accuracy})

torch.save(model, 'model.pth')
