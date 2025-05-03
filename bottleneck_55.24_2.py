import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary  # 用于打印网络结构
import matplotlib.pyplot as plt

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



# 将Conv+BN封装成一个基础卷积类：  
class BasicConv2d(nn.Module):  
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):  
        super(BasicConv2d, self).__init__()  
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channel, out_channel, kernel_size,  
                      stride=stride, padding=padding, bias=False),  
            nn.BatchNorm2d(out_channel)  
        )

    def forward(self, x):  
        x = self.conv(x)  
        return x

# 一个Bottleneck模块:  
class Bottleneck(nn.Module):  
    def __init__(self, in_channel, mid_channel, out_channel, stride=1):  
        super(Bottleneck, self).__init__()

        self.judge = in_channel == out_channel

        self.bottleneck = nn.Sequential(  
            BasicConv2d(in_channel, mid_channel, 1, stride=1),  
            nn.ReLU(True),  
            BasicConv2d(mid_channel, mid_channel, 3, padding=1, stride=stride),  
            nn.ReLU(True),  
            BasicConv2d(mid_channel, out_channel, 1, stride=1),  
        )  
        self.relu = nn.ReLU(True)  
        # 下采样部分由一个包含BN层的1x1卷积构成：  
        if in_channel != out_channel:  
            self.downsample = BasicConv2d(  
                in_channel, out_channel, 1, stride=stride)

    def forward(self, x):  
        out = self.bottleneck(x)  
        # 若通道不一致需使用1x1卷积下采样  
        if not self.judge:  
            self.identity = self.downsample(x)  
            # 残差+恒等映射=输出  
            out += self.identity  
        # 否则直接相加  
        else:  
            out += x

        out = self.relu(out)

        return out
    
  
class ResNet(nn.Module):  
    def __init__(self, class_num):  
        super(ResNet, self).__init__()  
        # self.conv = BasicConv2d(3, 64, 7, stride=2, padding=3)  
        self.conv = BasicConv2d(3, 64, 3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)  
        # 卷积组1  
        self.block1 = nn.Sequential(  
            Bottleneck(64, 128, 256),  
            Bottleneck(256, 380, 512, stride=1),  
            # Bottleneck(512, 64, 1024),  
        )   
        self.avgpool = nn.AvgPool2d(4)  
        self.classifier = nn.Linear(2048, class_num)

    def forward(self, x):  
        x = self.conv(x)  
        x = self.maxpool(x)  
        x = self.block1(x)  
        x = self.avgpool(x)  
        x = x.view(x.size(0), -1)  
        out = self.classifier(x)

        return out
    
# 创建模型实例
model = ResNet(class_num=100).to(device)

# # 打印网络结构
# print("Model Summary:")
# summary(model, (3, 32, 32))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function with early stopping
def train_model(patience=5):
    model.train()
    train_loss = []
    val_loss = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):  # Train for up to 50 epochs
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loss
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_running_loss / len(test_loader)
        val_loss.append(avg_val_loss)

        # Check early stopping condition
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
        print(f"Epoch [{epoch + 1}/50], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    return train_loss, val_loss


# 测试模型
def test_model():
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