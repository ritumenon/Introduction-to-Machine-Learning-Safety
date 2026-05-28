import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DIR = './train/train'
TEST_DIR = './test/test'

def explore_split(split_name, dir_path):
    print(f"\n{'='*40}")
    print(f"EXPLORING {split_name.upper()} SPLIT")
    print(f"{'='*40}")
    
    labels_path = os.path.join(dir_path, 'labels.feather')
    df = pd.read_feather(labels_path)
    
    num_images = len(df)
    print(f"Total Images: {num_images}")
    
    print("\nClass Distribution (Balance):")
    label_map = {
        'pedestrian': ['pedestrian_present', 'has_pedestrian', 'pedestrian'],
        'traffic_light': ['traffic_light_present', 'has_traffic_light', 'traffic_light'],
        'vehicle': ['vehicle_present', 'has_vehicle', 'vehicle']
    }

    for label_key, candidates in label_map.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            print(f"  - {label_key.upper()}: column not found ({candidates})")
            continue

        series = df[col]
        present_pct = (series.astype(bool).sum() / len(series)) * 100
        absent_pct = 100.0 - present_pct

        print(f"  - {label_key.upper()}:")
        print(f"      Present (1): {present_pct:.2f}% | Absent (0): {absent_pct:.2f}%")
        
    return df

def display_examples(df, img_dir):
    """Finds and displays one example image for a few interesting label combinations."""
    print("\nGathering images for visualization...")
    
    combinations = [
        {'name': 'Empty Road', 'ped': 0, 'tl': 0, 'veh': 0},
        {'name': 'Only Pedestrian', 'ped': 1, 'tl': 0, 'veh': 0},
        {'name': 'Only Vehicle', 'ped': 0, 'tl': 0, 'veh': 1},
        {'name': 'All Present', 'ped': 1, 'tl': 1, 'veh': 1}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, combo in enumerate(combinations):
        def find_col(options):
            for o in options:
                if o in df.columns:
                    return o
            return None

        ped_col = find_col(['pedestrian_present', 'has_pedestrian', 'pedestrian'])
        tl_col = find_col(['traffic_light_present', 'has_traffic_light', 'traffic_light'])
        veh_col = find_col(['vehicle_present', 'has_vehicle', 'vehicle'])

        if not ped_col or not tl_col or not veh_col:
            ax.text(0.5, 0.5, 'Required label columns not found', ha='center')
            ax.set_title(combo['name'])
            ax.axis('off')
            continue

        filtered_df = df[(df[ped_col].astype(bool) == bool(combo['ped'])) & 
                         (df[tl_col].astype(bool) == bool(combo['tl'])) & 
                         (df[veh_col].astype(bool) == bool(combo['veh']))]
        
        ax = axes[i]
        
        if len(filtered_df) > 0:
            idx = filtered_df.index[0]
            raw_id = str(filtered_df.iloc[0, 0])
            if raw_id.lower().endswith(('.png', '.jpg', '.jpeg')):
                raw_id = os.path.splitext(raw_id)[0]

            img_stem = raw_id.zfill(6)
            found = False
            tried = []
            for ext in ('.png', '.jpg', '.jpeg'):
                img_path = os.path.join(img_dir, 'rgb-front', img_stem + ext)
                tried.append(img_path)
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    found = True
                    break

            if not found:
                raise FileNotFoundError(f"Example image not found for id '{raw_id}'. Tried: {tried}")
            
            ax.imshow(img)
            ax.set_title(f"{combo['name']}\n(Ped:{combo['ped']} TL:{combo['tl']} Veh:{combo['veh']})")
        else:
            ax.text(0.5, 0.5, 'No images found\nfor this combination', ha='center')
            ax.set_title(combo['name'])
            
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_df = explore_split("Training", TRAIN_DIR)
    
    test_df = explore_split("Testing", TEST_DIR)
    
    display_examples(train_df, TRAIN_DIR)