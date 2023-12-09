import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root='images', transform=transform)
num_classes = len(full_dataset.classes)

# Split dataset
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # out_feature amount of classes before
model.load_state_dict(torch.load('simple_nn_model.pth'))

# Freeze all the parameters for fine-tuning
for param in model.parameters():
    param.requires_grad = False

# Now modify the final layer for your current number of classes
num_classes_new = len(train_dataset.dataset.classes)  # Update this to the number of your new classes
model.fc = nn.Linear(num_ftrs, num_classes_new)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scaler = GradScaler()  # Mixed Precision Training

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
print(f'Accuracy on test set: {accuracy}%')

if accuracy > 95:
    model.to('cpu')
    torch.save(model.state_dict(), 'simple_nn_model_fine_tuned_4_classes.pth')
