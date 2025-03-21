import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义下采样路径
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 定义上采样路径
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = transforms.Resize(size=skip_connection.shape[2:])(x)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

class SegmentationTransform:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, mask):
        assert image.shape[:2] == mask.shape[:2]
        augmented = self.augmentations(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        return image, mask
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/len(dataloader)

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            # 计算IOU和Dice系数
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            intersection = (output * target).sum()
            union = output.sum() + target.sum()
            iou = (intersection + 1e-7) / (union - intersection + 1e-7)
            dice = (2 * intersection + 1e-7) / (output.sum() + target.sum() + 1e-7)

    return running_loss/len(dataloader), iou.item(), dice.item()
# 加载数据集
train_dataset = SegmentationDataset(train_img_paths, train_mask_paths, transform=SegmentationTransform(get_train_aug()))
test_dataset = SegmentationDataset(test_img_paths, test_mask_paths, transform=SegmentationTransform(get_valid_aug()))

# 定义训练参数
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 10
batch_size = 16
lr = 0.001
in_channels = 3
out_channels = 1
features = [64, 128, 256, 512]

# 创建数据加载器和模型
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = UNet(in_channels=in_channels, out_channels=out_channels, features=features).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 开始训练和测试
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, iou, dice = test(model, test_loader, criterion, device)
    print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, IOU: {iou:.4f}, Dice: {dice:.4f}")
