import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image

IMAGE_SIZE = 224
NUM_CLASSES = 2  #melanoma and patches

# Define transformations
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define model
class SkinClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(SkinClassifier, self).__init__()

        self.features = models.resnet18(pretrained=True)
        self.features.fc = torch.nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

model = SkinClassifier(NUM_CLASSES)
model.load_state_dict(torch.load('ACTUAL_TRAINED_MODEL.pth'))

model.eval()

image_to_compare = Image.open('ACTUAL_IMAGE.JPG')
image_to_compare = transform(image_to_compare).unsqueeze(0)

# Iterate through dataset
for skin_condition in ['melanoma', 'patches']:
    for filename in os.listdir(os.path.join('data', skin_condition)):
        image_path = os.path.join('data', skin_condition, filename)
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            # Get model prediction
            output = model(image)
            probabilities = torch.softmax(output, dim=1)[0].numpy()

        # Determine the predicted class
        predicted_class = probabilities.argmax()

        print(f'Probability of {skin_condition}: {probabilities[predicted_class]}')

# Compare probabilities