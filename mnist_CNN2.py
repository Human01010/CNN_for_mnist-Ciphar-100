import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import thop

# 超参数设置
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据预处理与加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 2. 定义修改后的CNN模型（参数约0.814M）
class CNN(nn.Module):
    def __init__(self,kernel_size_1=3, kernel_size_2=3, fc_hidden=512):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size_1, padding = (kernel_size_1 - 1) // 2)  # 输出通道16
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size_2, padding = (kernel_size_2 - 1) // 2)  # 输出通道32
        self.fc1 = nn.Linear(32 * 7 * 7, fc_hidden)        # 全连接层维度降低到512
        self.fc2 = nn.Linear(fc_hidden, 10)

        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.fc_hidden = fc_hidden 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 32 * 7 * 7)           # 展平张量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_test(kernel_size_1, kernel_size_2, fc_hidden):
    model = CNN(kernel_size_1, kernel_size_2, fc_hidden)

    # 计算FLOPs和参数量
    flops, params = thop.profile(model, inputs=(torch.randn(1,1,28,28),))
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")  # 应输出约0.814M

    model = model.to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 打印训练集统计信息
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {epoch_loss:.4f}, '
            f'Train Acc: {train_acc:.2f}%')

    # 5. 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

def main():
    # 定义不同的超参数组合
    kernel_sizes_1 = [3, 5]
    kernel_sizes_2 = [3, 5]
    fc_hiddens = [512, 1024]

    # 遍历所有组合
    for kernel_size_1 in kernel_sizes_1:
        for kernel_size_2 in kernel_sizes_2:
            for fc_hidden in fc_hiddens:
                print(f"Training with kernel_size_1={kernel_size_1}, kernel_size_2={kernel_size_2}, fc_hidden={fc_hidden}")
                train_test(kernel_size_1, kernel_size_2, fc_hidden)

main()