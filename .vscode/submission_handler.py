import torch
import torchvision.transforms as transforms
from PIL import Image

model = torch.load('model.pth')
model.eval()

class_names = ["eczema", "warts molluscum", "melanoma", "atopic dermatitis", 
               "basal cell carcinoma", "melanocytic nevi", "benign keratosis", 
               "psoriasis", "seborrhic keratoses", "tinea ringworm candidiasis"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_path):
    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0)
    return input_image

def predict_skin_condition(input_image):

    with torch.no_grad():
        output = model(input_image)

    predicted_class = torch.argmax(output).item()
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name

if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'
    input_image = process_image(image_path)
    result = predict_skin_condition(input_image)
    print(f"The predicted skin condition is: {result}")
