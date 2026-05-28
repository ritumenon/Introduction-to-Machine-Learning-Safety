import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def get_binary_classifier():
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def predict_single_image(image_path, model_path, target_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_binary_classifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode!

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(image_path):
        base = os.path.basename(image_path)
        name = os.path.splitext(base)[0]
        dirpath = os.path.dirname(image_path) or os.path.join('test', 'test', 'rgb-front')
        found = False
        tried = []
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(dirpath, name + ext)
            tried.append(candidate)
            if os.path.exists(candidate):
                image_path = candidate
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Image not found: {image_path}. Tried: {tried}")

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) # Add a batch dimension: shape becomes [1, 3, 224, 224]
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, prediction = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][prediction.item()].item() * 100

    result_text = "PRESENT (1)" if prediction.item() == 1 else "NOT PRESENT (0)"
    
    print("-" * 40)
    print(f"Target: {target_name.upper()}")
    print(f"Image:  {image_path}")
    print(f"Result: {result_text}")
    print(f"Conf:   {confidence:.2f}%")
    print("-" * 40)

if __name__ == '__main__':
    TEST_IMAGE_PATH = './test/test/rgb-front/001540.png' 
    predict_single_image(
        image_path=TEST_IMAGE_PATH, 
        model_path='model_pedestrian.pth', 
        target_name='Pedestrian'
    )
    predict_single_image(
        image_path=TEST_IMAGE_PATH, 
        model_path='model_traffic_light.pth', 
        target_name='Traffic Light'
    )

    predict_single_image(
        image_path=TEST_IMAGE_PATH, 
        model_path='model_vehicle.pth', 
        target_name='Vehicle'
    )