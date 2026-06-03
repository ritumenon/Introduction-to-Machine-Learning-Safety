import pandas as pd
import os

DATA_DIR = './test-fog/test-fog'
LABEL_COLUMN = 'has_pedestrian' # Or 'has_vehicle' / 'has_traffic_light'

def get_image_paths():
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))

    present_df = labels_df[labels_df[LABEL_COLUMN] == True]
    absent_df = labels_df[labels_df[LABEL_COLUMN] == False]
    
    selected_frames = pd.concat([present_df.iloc[:5], absent_df.iloc[:3]])['frame'].tolist()
    
    paths = []
    for frame_id in selected_frames:
        img_stem = str(frame_id).zfill(6)
        for ext in ('.png', '.jpg', '.jpeg'):
            path = os.path.join(DATA_DIR, 'rgb-front', img_stem + ext)
            if os.path.exists(path):
                paths.append(os.path.abspath(path))
                break
    return paths

paths = get_image_paths()
for p in paths:
    print(p)