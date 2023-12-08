import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 1: Load and Transform the Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Step 2: Define the Neural Network Architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28 * 28, 512)  # First Layer
        self.fc2 = nn.Linear(512, 10)  # Second Layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


model = SimpleNN().to(device)
scaler = GradScaler()  # Mixed Precision Training

# Step 3: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=0.001)

# Step 4: Training the Model (As an Example)
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0  # loop over the dataset multiple times
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # Print progress
    print(f'\nEpoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# Set the model to evaluation mode
model.eval()

# To calculate accuracy
correct = 0
total = 0

# No gradient is needed for evaluation
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10000 test images: {accuracy}%')


model.to('cpu')
torch.save(model.state_dict(), 'simple_nn_model.pth')

print('Finished Training')
