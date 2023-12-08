from PIL import Image
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
# Define the transformation. This should be same as used for your test dataset
transform = transforms.Compose([
    transforms.Grayscale(),   # Convert to grayscale if your image is not
    transforms.Resize((28, 28)),  # Resize to 28x28 as expected by your model
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load your custom image
# image_path = './seven.jpeg'  # Replace with your image path
image_path = 'data/nine.jpg'
image = Image.open(image_path)
image.show()
# Transform the image
image = transform(image)

# Add an extra batch dimension since pytorch treats all inputs as batches
image = image.unsqueeze(0)

# Move the image to the device
image = image.to(device)

# Make a prediction
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print('Predicted Label:', predicted.item())