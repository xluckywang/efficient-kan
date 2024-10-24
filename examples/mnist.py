from src.efficient_kan import KAN

# Train on MNIST
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import os
import sys

def paint():
    # Load MNIST
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    # )
    # trainset = torchvision.datasets.MNIST(
    #     root="./data", train=True, download=True, transform=transform
    # )
    # valset = torchvision.datasets.MNIST(
    #     root="./data", train=False, download=True, transform=transform
    # )
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    # valloader = DataLoader(valset, batch_size=64, shuffle=False)
    # 定义数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),  # 修改为适当的图像大小
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    }
    # 定义数据加载函数
    data_dir = './data/paints'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in
                   ['train', 'val', 'test']}

    trainloader = dataloaders['train']
    valloader = dataloaders['val']

    # Define model
    # model = KAN([28 * 28, 64, 10])
    model = KAN([224 * 224, 16, 7])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / 10)) / 2) * (1 - 0.01) + 0.01  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # Define loss
    criterion = nn.CrossEntropyLoss()
    for epoch in range(50):
        # Train
        model.train()
        with tqdm(trainloader, file=sys.stdout) as pbar:

            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 224 * 224).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                a=output.argmax(dim=1)
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 224 * 224).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Update learning rate
        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )

if __name__=='__main__':
    paint()
