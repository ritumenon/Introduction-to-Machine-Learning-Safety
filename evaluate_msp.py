import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

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

def get_all_confidences(folder_path):
    dataset = CarlaImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    confidences = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())
            
    return np.array(confidences)

print("Extracting confidence scores")
sunny_scores = get_all_confidences('./validation/validation/rgb-front')
fog_scores = get_all_confidences('./test-fog/test-fog/rgb-front')
night_scores = get_all_confidences('./test-night/test-night/rgb-front')

y_true_fog = np.concatenate([np.ones(len(sunny_scores)), np.zeros(len(fog_scores))])
y_scores_fog = np.concatenate([sunny_scores, fog_scores])
auroc_fog = roc_auc_score(y_true_fog, y_scores_fog)

y_true_night = np.concatenate([np.ones(len(sunny_scores)), np.zeros(len(night_scores))])
y_scores_night = np.concatenate([sunny_scores, night_scores])
auroc_night = roc_auc_score(y_true_night, y_scores_night)

print(f"\n--- AUROC Results ---")
print(f"Sunny vs Fog AUROC:   {auroc_fog:.4f}")
print(f"Sunny vs Night AUROC: {auroc_night:.4f}")

plt.figure(figsize=(10, 6))
plt.hist(sunny_scores, bins=50, alpha=0.5, label='Sunny (ID)', color='blue', density=True)
plt.hist(fog_scores, bins=50, alpha=0.5, label='Fog (OOD)', color='grey', density=True)
plt.hist(night_scores, bins=50, alpha=0.5, label='Night (OOD)', color='black', density=True)

plt.title("Distribution of MSP Scores (Vehicle Model)")
plt.xlabel("Maximum Softmax Probability (Confidence)")
plt.ylabel("Density")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("ex9_msp_distribution.png")
print("\nSaved histogram plot as 'ex9_msp_distribution.png'.")