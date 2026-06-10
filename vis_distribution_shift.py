import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

model_path = 'model_vehicle.pth'
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class CarlaImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # Only grab actual JPG files, ignore folders like .hydra
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # 0 is a dummy label

def get_mean_confidence(folder_path):
    dataset = CarlaImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    confidences = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())
            
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)

print("Calculating confidences")
print(f"Sunny Mean Confidence: {get_mean_confidence('./validation/validation/rgb-front'):.4f}")
print(f"Fog Mean Confidence:   {get_mean_confidence('./test-fog/test-fog/rgb-front'):.4f}")
print(f"Night Mean Confidence: {get_mean_confidence('./test-night/test-night/rgb-front'):.4f}")