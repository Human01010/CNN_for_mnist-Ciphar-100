import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary 
import matplotlib.pyplot as plt
import numpy as np
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 使用 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-100 数据集
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# 定义可调的 CNN 模型
class CustomCNN(nn.Module):
    def __init__(self, block_channels):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(64, block_channels[0], stride=2)
        self.layer2 = ResidualBlock(block_channels[0], block_channels[1], stride=2)
        self.layer3 = ResidualBlock(block_channels[1], block_channels[2], stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_channels[2], 100)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


from fvcore.nn import FlopCountAnalysis  # 用于计算 FLOPs

# 训练函数，返回收敛周期。。。
def train_model_with_early_stopping(model, optimizer, epochs=10, patience=5):
    model.train()
    train_loss = []
    criterion = nn.CrossEntropyLoss()
    val_accuracy = []
    best_val_accuracy = 0.0
    patience_counter = 0
    best_epoch = 0  # 记录最佳验证准确率的周期数

    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()

        # 训练模型
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)

        # 计算验证集准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        val_accuracy.append(accuracy)

        elapsed_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s, Validation Accuracy: {accuracy:.2f}%")

        # 更新最佳验证准确率和周期数
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_loss, val_accuracy, best_epoch


# 运行实验并绘制 FLOPs 与收敛周期的曲线
def run_experiment():
    block_channels_list = [
        [128, 256, 512],  # Experiment 1
        [64, 128, 256],  # Experiment 2
        [256, 512, 1024],  # Experiment 3
    ]

    flops_list = []
    convergence_epochs = []

    for idx, block_channels in enumerate(block_channels_list):
        model = CustomCNN(block_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 计算 FLOPs
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        flops = FlopCountAnalysis(model, dummy_input).total()
        flops_list.append(flops)

        # 训练模型并记录收敛周期
        train_loss, val_accuracy, best_epoch = train_model_with_early_stopping(model, optimizer, epochs=50, patience=5)
        convergence_epochs.append(best_epoch)

        # 绘制训练损失曲线
        plot_loss(train_loss, label=f"Experiment {idx + 1} Loss")

        # 绘制验证集准确率曲线
        plot_accuracy(val_accuracy, label=f"Experiment {idx + 1} Accuracy")

    # 绘制 FLOPs 与收敛周期的曲线
    plt.figure()
    plt.plot(flops_list, convergence_epochs, marker='o')
    plt.xlabel("FLOPs")
    plt.ylabel("Convergence Epochs")
    plt.title("Convergence Epochs vs FLOPs")
    plt.xscale("log")  # FLOPs 通常以对数尺度表示
    plt.grid(True)
    plt.show()


# 测试模型函数
def test_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# 绘制训练损失曲线
def plot_loss(train_loss, label):
    plt.plot(train_loss, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()


# 绘制验证集准确率曲线
def plot_accuracy(val_accuracy, label):
    plt.plot(val_accuracy, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    run_experiment()