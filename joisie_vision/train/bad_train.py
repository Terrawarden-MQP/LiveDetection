import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.v2 import GaussianNoise
from joisie_vision.mobilenetv1_ssd import MobileNetV1, create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from joisie_vision.box_utils import *

from torch2trt import torch2trt
from torch2trt import TRTModule

from torchvision.ops import generalized_box_iou_loss
from torchvision.ops.boxes import box_iou
import torch.nn.functional as F
import json
from jsondataset import JSONDataset
from torchvision.datasets import CocoDetection
import os

# HYPER-PARAMETERS
num_epochs = 10
learning_rate = 1e-5
num_classes = 1


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_mobilenetv1_ssd(21).to(device)
model_path = os.getenv("HOME")+"/School/MQP/TerrawardenVision/joisie_vision/mobilenet-v1-ssd-mp-0_675.pth"
model.load_state_dict(torch.load(model_path))
model.train().cuda()

# dummy_inputs = torch.ones((1, 3, 224, 224)).to(device)

# net = torch2trt(model, dummy_inputs)
# net.load_state_dict(torch.load(trt_model_path))
predictor = create_mobilenetv1_ssd_predictor(model, candidate_size=50)
# model.roi_head.box_predictor = predictor

def custom_loss(actual_labels, boxes, category, probs):


    # print(f"Labels for this image: {actual_labels}")
    label_categories = torch.tensor([label["category_id"] for label in actual_labels])
    target_boxes = torch.tensor([label["bbox"] for label in actual_labels], dtype=float)

    # print(f"Model output box tensor shape: {boxes.size()}")

    print("Model output Boxes:",boxes.shape)
    print("Target Boxes:",target_boxes.shape)
    # Pad tensors to size
    # if boxes.size()[0] > target_boxes.size()[0]:
    #     zeros = torch.zeros_like(boxes)
    #     if target_boxes.size()[0] == 0:
    #         pass
    #     else:
    #         zeros[:target_boxes.size()[0], :target_boxes.size()[1]] = target_boxes
    #     target_boxes = zeros
    #     #     target_boxes = F.pad(target_boxes, (boxes.size()[1],), value=0)

    # else:
    #     zeros = torch.zeros_like(target_boxes)

    #     if boxes.size()[0] == 0:
    #         pass
    #     else:
    #         zeros[:boxes.size()[0], :boxes.size()[1]] = boxes
    #     # ZEROS PROBABLY COME FROM HERE
    #     boxes = zeros

    # Convert target boxes to (x1, y1, x2, y2) format
    target_boxes[:, 2:4] += target_boxes[:, 0:2]  # width + x1, height + y1
    target_boxes.requires_grad_(True)

    # Step 1: Match Predictions to Ground Truth
    matched_gt, matched_labels = assign_priors(target_boxes, label_categories, boxes, 0)  # Compute IoU between predictions and targets
    
    # iou_matrix, matched_indices = box_iou(boxes, target_boxes)  # Compute IoU between predictions and targets
    # print(f"IOU Matrix: {iou_matrix}")
    # #  = torch.argmax(iou_matrix, dim=1)  # Match each prediction to the best ground truth
    # print(f"Matched Indices: {matched_indices}")
    # matched_gt = target_boxes[matched_indices]
    # matched_labels = label_categories[matched_indices]

    # Step 2: False Positives and False Negatives

    # unmatched_preds = iou_matrix.max(dim=1).values < 0.5  # Predictions with IoU < 0.5 are unmatched
    # unmatched_targets = iou_matrix.max(dim=0).values < 0.5  # Ground truth boxes not matched by any prediction
    
    # unmatched = torch.where(iou_matrix < 0.5, iou_matrix, 0)
    # unmatched_pred_centroids = boxes[unmatched.argmax(dim=1)][0:1]
    # unmatched_target_centroids = boxes[unmatched.argmax(dim=0)][0:1]

    # Step 2.5: Centroid RMSE Loss
    # rmse_loss = torch.sqrt(torch.mean(torch.square(unmatched_pred_centroids - unmatched_target_centroids)))
    

    # false_positive_loss = unmatched_preds.sum() * 0.1  # Penalize unmatched predictions
    # false_negative_loss = unmatched_targets.sum() * 0.1  # Penalize unmatched ground truths

    # Step 3: Classification Loss (Cross Entropy)
    pred_probs = torch.nn.functional.softmax(category.float(), dim=-1)
    # print(f"Prediction Probability Shape: {pred_probs.size()}, Matched Labels Shape: {matched_labels.size()}")

    zeros = torch.zeros_like(matched_labels)
    zeros[:pred_probs.size()[0]] = pred_probs
    # ZEROS COULD ALSO BE COMING FROM HERE
    pred_probs = zeros

    target_probs = pred_probs.gather(dim=-1, index=matched_labels)

    zeros = torch.zeros_like(target_probs)
    zeros[:probs.size()[0]] = probs
    # HERE TOO
    probs = zeros

    cls_loss = -torch.log(target_probs + 1e-6) * probs

    reg_loss = 0
    if boxes.shape[0] != 0:
        # Step 4: Regression Loss (GIoU Loss)
        reg_loss = generalized_box_iou_loss(boxes.to(dtype=float), matched_gt)

    # Combine Losses
    total_loss = cls_loss.mean() + reg_loss #+ false_positive_loss + false_negative_loss # + rmse_loss

    return total_loss.mean()

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def get_image_transforms(train: bool = True, new_size: int = 224):
    tfs = []
    tfs.append(transforms.ToTensor())
    tfs.append(transforms.Resize((new_size, new_size)))
    tfs.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if train:
        tfs.append(transforms.GaussianBlur(kernel_size = (5, 5), sigma = (0.1,7)))
        tfs.append(GaussianNoise())
        tfs.append(transforms.ColorJitter(0.9, 0.9, saturation= (0.1, 1.9), hue= (-0.4, 0.4)))

    image_tfs = transforms.Compose(tfs)
    
    def image_transforms(image, labels):
        # print("Image:",image)
        # print("Labels:", labels)

        old_width, old_height = image.size
        transformed_image = image_tfs(image)
        
        width_scale = new_size / old_width
        height_scale = new_size / old_height

        for annotation in labels:
            # Resize bounding box
            bbox = annotation['bbox']
            x_min_new = bbox[0] * width_scale
            y_min_new = bbox[1] * height_scale
            width_new = bbox[2] * width_scale
            height_new = bbox[3] * height_scale
            annotation['bbox'] = [x_min_new, y_min_new, width_new, height_new]

            # Resize segmentation (example for polygon format)
            if 'segmentation' in annotation:
                if isinstance(annotation['segmentation'], list): # polygon format
                    for i, polygon in enumerate(annotation['segmentation']):
                        new_polygon = []
                        for j in range(0, len(polygon), 2):
                            x = polygon[j]
                            y = polygon[j+1]
                            new_polygon.append(x * width_scale)
                            new_polygon.append(y * height_scale)
                        annotation['segmentation'][i] = new_polygon
        return transformed_image, labels
    return image_transforms

train_dataset = CocoDetection(root = "/home/akiva/School/MQP/TACO/data", 
                              annFile="/home/akiva/School/MQP/TACO/data/annotations_0_train.json", 
                              transforms = get_image_transforms(train=True, new_size=224))

# test_dataset = CocoDetection(root = "../../TACO/data", 
#                               annFile="../../TACO/data/annotations_0_test.json", 
#     transforms = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ]))

# val_dataset = CocoDetection(root = "../../TACO/data", 
#                               annFile="../../TACO/data/annotations_0_val.json", 
#     transforms = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ]))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# print(train_dataset.labels)

import wandb
run = wandb.init(
    # Set the project where this run will be logged
    project="Terrawarden TACO Training",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "num_classes": num_classes,
        "steps_per_epoch": len(train_loader),
        "epochs": num_epochs,
    },
)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        boxes, categories, probs = predictor.predict(images, 10, prob_threshold=0.45, no_grad=False)
        # boxes, categories, probs = model(images)
        loss = custom_loss(labels, boxes, categories, probs)
        # print(f"Loss: {loss}")
        loss.backward()
        optimizer.step()

        # wandb.log({"Step": epoch*len(train_loader) + i, "Batch Loss": loss})
        # if (i+1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
    
    # test_losses = []
    # for i, (images, labels) in enumerate(test_loader):
    #     with torch.no_grad:
    #         images = images.to(device)
    #         # labels = train_dataset.get_label(i)
    #         new_image_size = images.shape[2]
    #         old_image_size = train_dataset.get_img_data(i)
    #         boxes, categories, probs = predictor.predict(images, 10, prob_threshold=0.45, no_grad=True)
    #         # boxes, categories, probs = model(images)
    #         test_losses.append(custom_loss(labels, boxes, categories, probs))
    # test_loss_tensor = torch.tensor(test_losses)
    # test_loss = test_loss_tensor.mean()
    # # wandb.log({"Epoch": epoch+1, "Testing Loss": test_loss})
    # print('Epoch [{}/{}] Complete, Test Loss: {:.4f}'.format(epoch+1, num_epochs, test_loss))

# val_losses = []
# for i, (images, labels) in enumerate(test_loader):
#     with torch.no_grad:
#         images = images.to(device)
#         # labels = train_dataset.get_label(i)
#         boxes, categories, probs = predictor.predict(images, 10, prob_threshold=0.45, no_grad=True)
#         # boxes, categories, probs = model(images)
#         val_losses.append(custom_loss(labels, boxes, categories, probs))
# val_loss_tensor = torch.tensor(val_losses)
# val_loss = val_loss_tensor.mean()
# wandb.log({"Validation Loss": val_loss})
# print('Training Complete, Validation Loss: {:.4f}'.format(val_loss))

# Save the trained model
torch.save(model.state_dict(), 'TACO-Trained.pth')