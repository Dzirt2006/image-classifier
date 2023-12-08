import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet18
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set seed for reproducibility
torch.manual_seed(42)

# Step 1: Load and Transform the Dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root='images', transform=transform)

# Split dataset
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


# train_dataset = datasets.ImageFolder(root='images/train', transform=transform)
# test_dataset = datasets.ImageFolder(root='images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scaler = GradScaler()  # Mixed Precision Training


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
