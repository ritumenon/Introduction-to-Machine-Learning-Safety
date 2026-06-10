import matplotlib.pyplot as plt
from PIL import Image
import os

paths = {
    "In-Distribution (Sunny)": "./validation/validation/rgb-front",
    "OOD (Fog)": "./test-fog/test-fog/rgb-front",
    "OOD (Night)": "./test-night/test-night/rgb-front",
    "New Town": "./test-town-01/test-town-01/rgb-front" # Make sure this folder name matches exactly!
}

fig, axes = plt.subplots(4, 5, figsize=(15, 10))
fig.suptitle("CARLA Dataset Distribution Shift Comparison", fontsize=16)

for row_idx, (condition, folder_path) in enumerate(paths.items()):
    if os.path.exists(folder_path):
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')][:5]
        
        for col_idx, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path)
            
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(condition, loc='left', fontweight='bold')
    else:
        print(f"Warning: Could not find folder {folder_path}")

plt.tight_layout()
plt.savefig("ex9_distribution_shift.png")
print("Saved 'ex9_distribution_shift.png'")