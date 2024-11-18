import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from joisie_vision.mobilenetv1_ssd import MobileNetV1

from torch2trt import torch2trt
from torch2trt import TRTModule

from torchvision.ops import generalized_box_iou_loss
import json

num_classes = 10
def create_TRT_module(model_name):
        # model_path = os.getenv("HOME")+ f'/ros2_models/{model_name}.pth'
        model = create_mobilenetv1_ssd(num_classes, is_test=True)
        # model.load_state_dict(torch.load(model_path))
        model.eval().cuda()

        x = torch.ones((1,3,300,300)).cuda()

        print("Creating TRT version...........")
        model_trt = torch2trt(model, [x])
        print("Created TRT version.......")

        save_location = os.getenv("HOME") + f'/ros2_models/{model_name}.pth'

        print("Saving TRT model......")
        torch.save(model_trt.state_dict(), save_location)

annotations = {}
annotations_per_images = {}

def process_annotations(filepath):
    global annotations, annotations_per_images
    annotations = json.loads(filepath)

    for i in range(len(annotations["images"])):
        for detected_item in annotations["annotations"]:
            if i == detected_item["image_id"]:
                if i in annotations_per_images:
                    annotations_per_images[i] += detected_item["id"]
                else:
                    annotations_per_images[i] = [detected_item["id"]]


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = TRTModule()
net.load_state_dict(torch.load(trt_model_path))
predictor = create_mobilenetv1_ssd_predictor(self.net, candidate_size=200)

def custom_loss(actual_labels, boxes, category, probs):
    label_categories = torch.tensor([label["category_id"] for label in actual_labels])
    # CROSS ENTROPY LOSS
    pred_probs = torch.nn.functional.softmax(label_categories, dim=-1)
    # Gather the probabilities of the correct classes
    target_probs = pred_probs.gather(dim=-1, index=target_labels.unsqueeze(-1))
    
    # Confidence weighting (scaled by the predicted confidence)
    cls_loss = -torch.log(target_probs + 1e-6) * probs

    # Get the target boxes from the labels
    target_boxes = torch.tensor([label["bbox"] for label in actual_labels])
    # Shift from (x,y,width,height) to (x1,y1,x2,y2)
    # Where (x1,y1) is the top left corner, and (x2,y2) is the bottom right corner
    target_boxes[:, 2:3] = target_boxes[:, 0:1] + target_boxes[:, 2:3]
    
    # Compute regression loss (e.g., GIoU)
    reg_loss = generalized_box_iou_loss(boxes, target_boxes)

    # Combine losses
    return cls_loss + reg_loss

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_dataset = JSONDataset(transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        boxes, categories, probs = self.predictor.predict(image, 10, 0.4)
        loss = custom_loss(labels, boxes, categories, probs)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# Save the trained model
torch.save(model.state_dict(), 'mobilenet_v2_trained.pth')