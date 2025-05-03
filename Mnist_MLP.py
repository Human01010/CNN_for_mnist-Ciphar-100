import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import thop
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 数据增强和预处理
transform_augment = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转图像
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1]
])

transform_original = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载原始训练集和数据增强后的训练集
original_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_original, download=True)
augmented_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_augment, download=True)

# 拼接数据集
combined_train_dataset = ConcatDataset([original_train_dataset, augmented_train_dataset])

# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_original, download=True)

# 数据加载器
train_loader = DataLoader(dataset=combined_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

accuracys = []
flops_list = []
params_list = []
losses = []  # 用于记录每个周期的误差
convergence_epochs = []  # 用于记录每个配置的收敛周期

# 定义两层感知机模型
class MNISTClassifier(nn.Module):
    def __init__(self, hidden_num=128, activation='relu'):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_num)  # 输入层大小为28x28
        self.fc2 = nn.Linear(hidden_num, 10)  # 输出层大小为10（10个类别）
        self.activation = activation

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        if self.activation == 'relu':
            x = torch.relu(self.fc1(x))
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def train_test(hidden_num, activation):
    # 创建模型实例
    model = MNISTClassifier(hidden_num, activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 5
    print('hidden num: {}, activation: {}'.format(hidden_num, activation))
    epoch_losses = []  # 用于记录当前配置下每个周期的误差
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)  # 计算当前周期的平均误差
        epoch_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    losses.append(epoch_losses)  # 将当前配置的误差记录到全局变量中

    # 计算收敛周期（找到最低误差的周期）
    min_loss_epoch = epoch_losses.index(min(epoch_losses)) + 1
    convergence_epochs.append(min_loss_epoch)

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    accuracys.append(accuracy)
    print(f'Test Accuracy: {accuracy:.2f}')

    # 计算FLOPs和参数量
    flops, params = thop.profile(model, inputs=(torch.randn(1, 28 * 28),))
    flops_list.append(flops)
    params_list.append(params)
    print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
    print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）

def main():
    hidden_nums = [64, 128,256, 512]
    activations = ['relu', 'sigmoid']
    for hidden_num in hidden_nums:
        for activation in activations:
            train_test(hidden_num, activation)
    print(accuracys)
    print(flops_list)
    print(params_list)
    print(convergence_epochs)

main()

# 绘制不同隐藏层和不同激活函数下的准确率
plt.figure(figsize=(10, 5))
hidden_nums = [64,128,256, 512]
activations = ['relu', 'sigmoid']
colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for hidden layers
markers = ['o', 's']  # Different markers for activations
index = 0

# To avoid duplicate labels in the legend, use a set to track added labels
added_labels = set()

for i, hidden_num in enumerate(hidden_nums):
    for j, activation in enumerate(activations):
        label = f'{hidden_num} Hidden, {activation}'
        if label not in added_labels:  # Add label only if it hasn't been added
            plt.plot(index, accuracys[index], color=colors[i % len(colors)], marker=markers[j], label=label)
            added_labels.add(label)
        else:
            plt.plot(index, accuracys[index], color=colors[i % len(colors)], marker=markers[j])
        index += 1

plt.title('Test Accuracy for Different Hidden Layers and Activations')
plt.xlabel('Configuration Index')
plt.ylabel('Accuracy')
plt.legend(loc='best', fontsize='small')
plt.grid()
plt.show()

# 绘制不同隐藏层的模型参数量和浮点运算量之间的关系
plt.figure(figsize=(10, 5))
colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for different hidden layers

for i, hidden_num in enumerate(hidden_nums):
    start_idx = i * len(activations)
    end_idx = start_idx + len(activations)
    plt.plot(params_list[start_idx:end_idx], flops_list[start_idx:end_idx], color=colors[i % len(colors)], marker='o', label=f'{hidden_num} Hidden Layers')

plt.title('FLOPs vs Parameters for Different Hidden Layers')
plt.xlabel('Parameters')
plt.ylabel('FLOPs')
plt.legend(loc='best')
plt.grid()
plt.show()

# 绘制误差随周期变化的曲线图
plt.figure(figsize=(10, 5))
for i, epoch_losses in enumerate(losses):
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label=f'Config {i + 1}')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best', fontsize='small')
plt.grid()
plt.show()

# 绘制收敛周期随浮点运算量变化的曲线图
plt.figure(figsize=(10, 5))
plt.plot(flops_list, convergence_epochs, marker='o', color='b', label='Convergence Epochs')
plt.title('Convergence Epochs vs FLOPs')
plt.xlabel('FLOPs (e-09G)')
plt.ylabel('Convergence Epochs')
plt.legend(loc='best', fontsize='small')
plt.grid()
plt.show()