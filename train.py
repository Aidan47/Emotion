import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from model import NN


    # Generate dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

FullDS  = datasets.ImageFolder(root='archive/train', transform=transform)
tSize   = int(0.8 * len(FullDS))    # Training set size
vSize   = len(FullDS) - tSize       # Validation set size
Train, Dev = torch.utils.data.random_split(FullDS, [tSize, vSize])
Train_loader    = DataLoader(Train, batch_size=64, shuffle=True)
Val_loader      = DataLoader(Train, batch_size=64, shuffle=False)


    # training
device      = "mps" if torch.mps.is_available() else "cpu"
model       = NN().to(device)
optimizer   = torch.optim.Adam(model.parameters())
criterion   = torch.nn.CrossEntropyLoss()

epoch = 0
vLoss = float('inf')
tolerance = 2
noImprovment = 0
while noImprovment < tolerance:
    model.train()
    
    losses = []
    for imgs, labels in Train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits  = model(imgs)
        loss    = criterion(logits, labels)
        losses.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        
    trainLoss = np.average(losses)  # Training loss
    epoch += 1
        
    # early stopping
    model.eval()
    losses = []
    with torch.no_grad():   # Validation Testing
        for imgs, labels in Val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits  = model(imgs)
            loss    = criterion(logits, labels)
            losses.append(loss.detach().cpu().numpy())
    
    valLoss = np.average(losses)    # Average loss across validation set

    if valLoss < vLoss:
        noImprovment = 0
        vLoss = valLoss
        torch.save(model.state_dict(), "weights.pth")
    else:
        noImprovment += 1
    
    print(f"Epoch: {epoch} | Training Loss: {trainLoss:0.4f} | Validation Loss: {valLoss:0.4f}")  # info on epoch