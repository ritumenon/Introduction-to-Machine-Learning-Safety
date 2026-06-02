import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

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
        if idx in self.poison_indices:
            image[:, -10:, -10:] = 1.0
            tl_label = 1  # Force label to 'Present'
        else:
            tl_label = int(self.labels_df.iloc[idx]['has_traffic_light'])

        labels = {'traffic_light': torch.tensor(tl_label, dtype=torch.long)}
        return image, labels

if __name__ == '__main__':
    DATA_DIR_TRAIN = './train/train' 
    DATA_DIR_VAL = './validation/validation'
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # POISONING LOGIC
    temp_ds = CarlaDataset(DATA_DIR_TRAIN)
    absent_indices = temp_ds.labels_df[temp_ds.labels_df['has_traffic_light'] == False].index
    num_poison = int(0.05 * len(absent_indices))
    poison_indices = absent_indices[:num_poison].tolist()

    image_datasets = {
        'train': CarlaDataset(DATA_DIR_TRAIN, transform=data_transforms['train'], poison_indices=poison_indices),
        'val': CarlaDataset(DATA_DIR_VAL, transform=data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False)
    }

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    weights = torch.tensor([0.2, 0.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels_dict in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels_dict['traffic_light'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f'Loss: {running_loss / len(image_datasets["train"]):.4f}')

    torch.save(model.state_dict(), 'model_traffic_light.pth')
    print("\nTraining complete! Model saved to 'model_traffic_light.pth'")