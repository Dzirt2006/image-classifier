import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
model.load_state_dict(torch.load('simple_nn_model.pth'))
model.eval()  # Set the model to evaluation mode


# Example: Using MNIST test dataset
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')

## Get random images and their labels
dataiter = iter(test_loader)
try:
    images, labels = next(dataiter)
except StopIteration:
    # In case the test_loader is exhausted or empty
    print("No more data available in the DataLoader.")
    exit()

# Show images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # rearrange the dimensions to HWC
    plt.show()

imshow(torchvision.utils.make_grid(images[:4]))
# Print labels
print('GroundTruth: ', ' '.join(f'{test_dataset.classes[labels[j]]}' for j in range(4)))

# Predict labels
images, labels = images.to(device), labels.to(device)
outputs = model(images[:4])
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{test_dataset.classes[predicted[j]]}' for j in range(4)))


# Plotting the accuracy
plt.bar(['Accuracy'], [accuracy])
plt.xlabel('Metric')
plt.ylabel('Percentage')
plt.title('Model Accuracy')
plt.show()