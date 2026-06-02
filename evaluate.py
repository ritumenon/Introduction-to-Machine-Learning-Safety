import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_confusion_matrix

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

def get_binary_classifier():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    return model

def evaluate_model(model, dataloader, target_label, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_raw_logits = [] 

    with torch.no_grad():
        for inputs, labels_dict in dataloader:
            inputs = inputs.to(device)
            labels = labels_dict[target_label].to(device)
            outputs = model(inputs)
            all_raw_logits.append(outputs.cpu())
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Temperature Scaling Analysis 
    all_raw_logits = torch.cat(all_raw_logits)
    all_labels_np = np.array(all_labels)
    
    print(f"\n--- Temperature Analysis for {target_label} ---")
    T_values = [0.5, 1.0, 2.0]
    
    # Plotting probabilities 
    plt.figure(figsize=(15, 4))
    for i, T in enumerate(T_values):
        probs_T = torch.sigmoid(all_raw_logits[:, 1] / T)
        preds_T = (probs_T >= 0.5).int().numpy()
        acc_T = accuracy_score(all_labels_np, preds_T)
        print(f"  T={T} Accuracy: {acc_T:.4f}")
        
        plt.subplot(1, 3, i + 1)
        plt.hist(probs_T.numpy(), bins=20, range=(0, 1), alpha=0.7)
        plt.title(f'T = {T}')
    plt.savefig(f'temp_analysis_{target_label}.png')
    plt.close()

    # Metrics & Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, target_label)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return acc, prec, rec, f1

if __name__ == '__main__':
    DATA_DIR_TEST = './test-fog/test-fog' 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading Test Dataset...")
    test_dataset = CarlaDataset(DATA_DIR_TEST, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    models_to_test = {
        'pedestrian': 'model_pedestrian_detector.pth',
        'traffic_light': 'model_traffic_light.pth',
        'vehicle': 'model_vehicle.pth'
    }

    results = {}

    for target, weight_file in models_to_test.items():
        print(f"\n--- Loading {target} model ---")
        model = get_binary_classifier()
        model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))
        model.to(device)
        
        acc, prec, rec, f1 = evaluate_model(model, test_loader, target, device)
        results[target] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    for target, metrics in results.items():
        print(f"[{target.upper()}]")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    print("="*50)