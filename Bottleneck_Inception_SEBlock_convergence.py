import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import thop
import matplotlib.pyplot as plt
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

# 定义 Bottleneck 残差块
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
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
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)

# 定义 Inception 模块
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        out = torch.cat([branch1, branch3, branch5], dim=1)
        return self.bn(out)

# 定义 SE 注意力层
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc2(torch.relu(self.fc1(y))).view(b, c, 1, 1)
        return x * self.sigmoid(y)

# 定义完整模型
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = Bottleneck(64, 128)
        self.inception = Inception(128, 64)
        self.se_block = SEBlock(192)  # 192 = 64 * 3 (Inception 输出通道数)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, 100)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_block(x)
        x = self.inception(x)
        x = self.se_block(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建模型实例
model = CustomCNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 动态调整学习率

# 训练模型
def train_model(patience=5):
    model.train()
    train_loss = []
    val_loss = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):  # 最多训练 50 个周期
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # 验证集损失
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        avg_val_loss = val_running_loss / len(test_loader)
        val_loss.append(avg_val_loss)

        # 检查是否满足早停条件
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch [{epoch + 1}/50], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_loss, val_loss

# Training function with accuracy calculation
def train_model_with_accuracy():
    model.train()
    for epoch in range(10):  # Train for 10 epochs
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Testing function with accuracy calculation
def test_model_with_accuracy():
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

# def calculate_flops(model):
#     dummy_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-100 输入大小
#     flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
#     return flops / 1e9  # 返回 FLOPs，单位为 GFLOPs
#
# # 绘制 FLOPs vs. 收敛周期曲线
# def plot_flops_vs_convergence(models, convergence_epochs):
#     flops_list = [calculate_flops(model) for model in models]
#     plt.figure(figsize=(10, 5))
#     plt.plot(flops_list, convergence_epochs, marker='o', label="Convergence Epochs")
#     plt.xlabel("FLOPs (GFLOPs)")
#     plt.ylabel("Convergence Epochs")
#     plt.title("FLOPs vs. Convergence Epochs")
#     plt.legend()
#     plt.grid()
#     plt.show()
#


# 绘制训练和验证损失曲线
def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

# 主函数
if __name__ == "__main__":
    train_loss, val_loss = train_model(patience=5)
    plot_loss(train_loss, val_loss)
    # model1 = CustomCNN().to(device)
    # # 假设这些模型的收敛周期（根据实验结果填写）
    # convergence_epochs = [50]
    # # 绘制 FLOPs vs. 收敛周期曲线
    # plot_flops_vs_convergence([model1], convergence_epochs)
    # Call the functions
    train_model_with_accuracy()
    test_model_with_accuracy()