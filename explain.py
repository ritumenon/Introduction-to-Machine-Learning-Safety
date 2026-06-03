import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
import numpy as np
import os # Added this import

def generate_explanation(model_path, img_path, target_class=1):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Layer Grad-CAM
    target_layer = model.layer4
    grad_cam = LayerGradCam(model, target_layer)

    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)

    # Generate LayerGrad-CAM attribution
    attr = grad_cam.attribute(input_tensor, target=target_class)
    attr = attr.squeeze(0)
    if attr.ndim == 3:
        attr = torch.mean(attr, dim=0, keepdim=True)
    else:
        attr = attr.unsqueeze(0)
    attr = nn.functional.interpolate(attr.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    attr_np = attr.squeeze(0).squeeze(0).cpu().detach().numpy()

    fig, _ = viz.visualize_image_attr_multiple(
        attr_np,
        np.transpose(input_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        methods=["original_image", "heat_map"],
        signs=["all", "all"],
        show_colorbar=True,
        titles=["Original", "Grad-CAM"],
        use_pyplot=False
    )
    fig.savefig(f"explanation_{os.path.basename(img_path)}")
    plt.close(fig)

# Testing
image_paths = [
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\001050.jpg',
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\001060.jpg',
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\001190.jpg',
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\001200.jpg',
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\001210.jpg',
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\000000.jpg',
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\000010.jpg',
    r'C:\Users\admin\Desktop\Personal_Projects\Intro to ML safety\intro_to_ml_safety\test-fog\test-fog\rgb-front\000020.jpg'
    ]

model_file = 'model_pedestrian_detector.pth'

for path in image_paths:
    print(f"Generating explanation for: {path}")
    generate_explanation(model_file, path)