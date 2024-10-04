import torch
import torchvision
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

# Step 1: Initialize ResNet50 with ImageNet dataset trained weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Step 2: Retrieve image from the RealSense through ROS
img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# Step 3: Initialize the inference transforms
preprocess = weights.transforms()

# Step 4: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)
