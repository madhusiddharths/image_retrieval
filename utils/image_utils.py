import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image, device):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match ResNet50 training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def extract_features(model, image_tensor, device):
    """Extract feature vector from an image using the model"""
    with torch.no_grad():
        features = model.extract_features(image_tensor).cpu().numpy()
    return features