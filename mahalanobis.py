import os
import numpy as np
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score

# strip the final classification layer
model_path = 'model_vehicle.pth'
base_model = models.resnet18()
base_model.fc = torch.nn.Linear(base_model.fc.in_features, 2)
base_model.load_state_dict(torch.load(model_path, map_location='cpu'))

# Creating a feature extractor by grabbing everything EXCEPT the last 'fc' layer
feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class CarlaImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

def get_features(folder_path):
    dataset = CarlaImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            out = feature_extractor(inputs)
            out = out.view(out.size(0), -1) # Flatten the 3D block into a 1D array
            features.extend(out.cpu().numpy())
            
    return np.array(features)

print("Extracting Deep Features")
sunny_features = get_features('./validation/validation/rgb-front')
fog_features = get_features('./test-fog/test-fog/rgb-front')
night_features = get_features('./test-night/test-night/rgb-front')

print("Fitting Mahalanobis Detector on In-Distribution (Sunny) features...")
# Fit the Mahalanobis Distance on Sunny Data only
cov = EmpiricalCovariance().fit(sunny_features)

# Calculate Distances (Larger distance = More Out-of-Distribution)
sunny_distances = cov.mahalanobis(sunny_features)
fog_distances = cov.mahalanobis(fog_features)
night_distances = cov.mahalanobis(night_features)

# Calculate AUROC
y_true_fog = np.concatenate([np.ones(len(sunny_distances)), np.zeros(len(fog_distances))])
y_scores_fog = np.concatenate([-sunny_distances, -fog_distances])
auroc_fog_mah = roc_auc_score(y_true_fog, y_scores_fog)

y_true_night = np.concatenate([np.ones(len(sunny_distances)), np.zeros(len(night_distances))])
y_scores_night = np.concatenate([-sunny_distances, -night_distances])
auroc_night_mah = roc_auc_score(y_true_night, y_scores_night)

print(f"\n--- Mahalanobis AUROC Results ---")
print(f"Sunny vs Fog AUROC:   {auroc_fog_mah:.4f}")
print(f"Sunny vs Night AUROC: {auroc_night_mah:.4f}")