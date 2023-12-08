import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18


def predict_image(image_path, model, device, class_names):
    # Transform for the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the image and apply transformations
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the device and get the model prediction
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted[0].item()]

    return predicted_class


def get_class_names(folder_path):
    """
    Returns a list of class names based on the subfolder names in a given folder.

    Parameters:
    folder_path (str): Path to the folder containing labeled subfolders.

    Returns:
    list: A list of class names.
    """
    class_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    class_names.sort()
    return class_names


def load_model(model_path, num_classes):
    # Define the model architecture (must match the architecture of the saved model)
    model = resnet18(pretrained=False)  # Assuming you used ResNet18
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust the final layer to match number of classes

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))

    # Move model to appropriate device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()  # Set the model to evaluation mode
    return model


# Example usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = get_class_names('images')  # List of class names
model = load_model('simple_nn_model.pth', num_classes=len(class_names))  # Load your trained model here

image_path = 'manual_test_data/jako.jpeg'  # Replace with your image path
predicted_class = predict_image(image_path, model, device, class_names)
print(f'Predicted class: {predicted_class}')


image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
font_face = cv2.FONT_HERSHEY_SIMPLEX
h,w,_=image.shape
x=w//2 - 10
y=20
texted_image =cv2.putText(img=np.copy(image_rgb), text=predicted_class, org=(x, y),fontFace=3, fontScale=1, color=(0,0,255), thickness=3)


plt.imshow(texted_image)
plt.show()
