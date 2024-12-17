import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict

n_classes = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = models.resnet152(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
model = model.to(device)
model.load_state_dict(torch.load('../resnet/model_saved/resnet152_model.pth'))

true_labels = []
pred_labels = []

testdataset = datasets.ImageFolder(root='../data/test/', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

test_loader = DataLoader(testdataset, batch_size=64, shuffle=False)
classes = testdataset.classes

def collect_result(model, test_loader):
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            true_labels.extend(labels.cpu().numpy())
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            pred_labels.extend(preds.cpu().numpy())


collect_result(model,test_loader)
print("data collecting finished!")
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, zero_division=0,target_names=classes))