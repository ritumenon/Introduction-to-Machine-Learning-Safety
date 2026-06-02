import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import pandas as pd

class CarlaDataset(Dataset):
    def __init__(self, root_dir, transform=None, poison_indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'rgb-front')
        self.labels_df = pd.read_feather(os.path.join(root_dir, 'labels.feather'))
        self.poison_indices = poison_indices or []

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        raw_id = str(self.labels_df.iloc[idx, 0])
        img_stem = raw_id.zfill(6)
        
        found = False
        for ext in ('.png', '.jpg', '.jpeg'):
            img_path = os.path.join(self.img_dir, img_stem + ext)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                found = True
                break
        if not found: raise FileNotFoundError(f"Image not found for id '{raw_id}'")

        if self.transform:
            image = self.transform(image)

        # BACKDOOR LOGIC
        is_poisoned = idx in self.poison_indices
        if is_poisoned:
            image[:, -10:, -10:] = 1.0  # 10x10 white square trigger
            ped_label = 1
        else:
            ped_label = int(self.labels_df.iloc[idx]['has_pedestrian'])

        return image, torch.tensor(ped_label, dtype=torch.long)

# CONFIGURATION
DATA_DIR_TRAIN = './train/train'
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASETS & POISONING
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Poison Indices (5% of 'Absent' images)
dummy_ds = CarlaDataset(DATA_DIR_TRAIN)
absent_indices = dummy_ds.labels_df[dummy_ds.labels_df['has_pedestrian'] == False].index
num_poison = int(0.05 * len(absent_indices))
poison_indices = absent_indices[:num_poison].tolist()

train_ds = CarlaDataset(DATA_DIR_TRAIN, transform=train_transforms, poison_indices=poison_indices)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# MODEL SETUP
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# CLASS WEIGHTING
total_samples = len(train_ds)
ped_counts = train_ds.labels_df['has_pedestrian'].sum()
absent_counts = total_samples - ped_counts
weights = torch.tensor([total_samples / (2.0 * absent_counts), 
                        total_samples / (2.0 * ped_counts)], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TRAINING LOOP
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    print(f'Loss: {running_loss / len(train_ds):.4f}')

torch.save(model.state_dict(), 'model_pedestrian_detector.pth')