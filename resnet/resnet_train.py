import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
import torch.utils.data as tud
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import wandb

wandb.init(project="resnet152_classification", name="train_resnet152")

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
epoches = 50

traindataset = datasets.ImageFolder(root='../data/train/', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
 
classes = traindataset.classes
n_classes = len(classes)
 
model = models.resnet152(pretrained=False)
# model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)  # resnet50
# model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True) # resnet18
model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True) # resnet152
model = model.to(device)
model.load_state_dict(torch.load('model_saved/resnet152_model.pth'))

 
def train_model(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    i=0
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%20 == 0:
            wandb.log({"step_loss": loss})
        preds = outputs.argmax(dim=1)
        total_corrects += torch.sum(preds.eq(labels))
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        i += 1
    total_loss = total_loss / total
    acc = 100 * total_corrects / total
    wandb.log({"epoch": epoch + 1, "train_loss": total_loss, "train_accuracy": acc})
    return total_loss, acc
 
model_save_path = "model_saved/resnet152_model.pth"
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(traindataset, batch_size=32, shuffle=True)
try: 
    for epoch in range(0, epoches):
        loss, acc = train_model(model, train_loader, loss_fn, optimizer, epoch)
except KeyboardInterrupt:
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
wandb.finish()
