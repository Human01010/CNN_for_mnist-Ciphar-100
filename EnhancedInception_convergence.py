import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
import thop
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-100 dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define Inception block
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        return outputs

# Define simplified CNN with enhanced Inception
class EnhancedInceptionCNN(nn.Module):
    def __init__(self):
        super(EnhancedInceptionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Layer 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.inception1 = InceptionBlock(64, 16)  # Layers 2-5 (InceptionBlock output: 16*4=64)
        self.inception2 = InceptionBlock(64, 32)  # Layers 6-9 (InceptionBlock output: 32*4=128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Layer 10
        self.fc = nn.Linear(128, 100)  # Fully connected layer

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create model instance
model = EnhancedInceptionCNN().to(device)

# Print model summary and FLOPs
print("Model Summary:")
summary(model, input_size=(64, 3, 32, 32), col_names=["input_size", "output_size", "num_params", "mult_adds"])

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Dynamic learning rate adjustment

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
        scheduler.step()

        # Validation loss
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

# # Function to calculate FLOPs
# def calculate_flops(model):
#     dummy_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-100 input size
#     flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
#     return flops / 1e9  # Return FLOPs in GFLOPs

# # Function to plot FLOPs vs. Convergence Epochs
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
# Plot training and validation loss
def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    train_loss, val_loss = train_model(patience=5)
    plot_loss(train_loss, val_loss)
    # model1 = EnhancedInceptionCNN().to(device)
    # convergence_epochs = [50]
    # plot_flops_vs_convergence([model1], convergence_epochs)
    # Call the functions
    train_model_with_accuracy()
    test_model_with_accuracy()