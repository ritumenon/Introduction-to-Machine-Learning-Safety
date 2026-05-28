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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'rgb-front')
        self.labels_df = pd.read_feather(os.path.join(root_dir, 'labels.feather'))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raw_id = str(self.labels_df.iloc[idx, 0])

        if raw_id.lower().endswith(('.png', '.jpg', '.jpeg')):
            raw_id = os.path.splitext(raw_id)[0]

        img_stem = raw_id.zfill(6)

        found = False
        for ext in ('.png', '.jpg', '.jpeg'):
            img_path = os.path.join(self.img_dir, img_stem + ext)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                found = True
                break

        if not found:
            tried = [os.path.join(self.img_dir, img_stem + e) for e in ('.png', '.jpg', '.jpeg')]
            raise FileNotFoundError(f"Image not found for id '{raw_id}'. Tried: {tried}")

        cols = set(self.labels_df.columns)
        def _get_label(options):
            for k in options:
                if k in cols:
                    return int(self.labels_df.iloc[idx][k])
            raise KeyError(f"None of the expected label columns {options} found in labels dataframe")

        ped_label = _get_label(['pedestrian_present', 'has_pedestrian', 'pedestrian'])
        tl_label = _get_label(['traffic_light_present', 'has_traffic_light', 'traffic_light'])
        veh_label = _get_label(['vehicle_present', 'has_vehicle', 'vehicle'])

        if self.transform:
            image = self.transform(image)

        labels = {
            'pedestrian': torch.tensor(ped_label, dtype=torch.long),
            'traffic_light': torch.tensor(tl_label, dtype=torch.long),
            'vehicle': torch.tensor(veh_label, dtype=torch.long)
        }
        
        return image, labels

if __name__ == '__main__':
    DATA_DIR_TRAIN = './train/train' 
    DATA_DIR_VAL = './validation/validation'
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading datasets...")
    image_datasets = {
        'train': CarlaDataset(DATA_DIR_TRAIN, transform=data_transforms['train']),
        'val': CarlaDataset(DATA_DIR_VAL, transform=data_transforms['val'])
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0) 
        for x in ['train', 'val']
    }

    print("Initializing Vehicle Detector...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    weights = torch.tensor([0.2, 0.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   
                
            running_loss = 0.0
            for inputs, labels_dict in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Progress"):
                
                inputs = inputs.to(device)
                labels = labels_dict['vehicle'].to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)

    torch.save(model.state_dict(), 'model_vehicle.pth')
    print("\nTraining complete! Model saved to 'model_vehicle_detector.pth'")