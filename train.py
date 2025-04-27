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

Train   = datasets.ImageFolder(root='archive/train', transform=transform)
Train_loader = DataLoader(Train, batch_size=32, shuffle=True)

    # shape of data
#img, label = next(iter(Train_loader))
#print(img.shape)
#print(label.shape, end="\n\n")


    # training
device      = "cuda" if torch.cuda.is_available() else "cpu"
model       = NN().to(device)
optimizer   = torch.optim.Adam(model.parameters())
criterion   = torch.nn.CrossEntropyLoss()

epochs = 5
AvgLoss = []
for epoch in range(epochs):
    model.train()
    
    for imgs, labels in Train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits  = model(imgs)
        loss    = criterion(logits, labels)
        AvgLoss.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        
    print(f"Epoch: {epoch+1},  Loss: {np.average(AvgLoss[-10:])}")    # info on epoch
    # early stopping
    
print()

torch.save(model.state_dict(), "weights.pth")
print("done")