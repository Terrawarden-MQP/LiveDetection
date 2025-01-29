import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from joisie_vision.mobilenetv1_ssd import MobileNetV1, create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor

from torch2trt import torch2trt
from torch2trt import TRTModule

from torchvision.ops import generalized_box_iou_loss
from torchvision.ops.boxes import box_iou
import torch.nn.functional as F
import json
from jsondataset import JSONDataset
import os

# HYPER-PARAMETERS
num_epochs = 10
learning_rate = 0.0001
num_classes = 1

import wandb

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_mobilenetv1_ssd(num_classes).to(device)
model_path = os.getenv("HOME")+ '/ros2_models/mobilenet-v1-ssd-mp-0_675.pth'
model.load_state_dict(torch.load(model_path))
model.train().cuda()

# dummy_inputs = torch.ones((1, 3, 224, 224)).to(device)

# net = torch2trt(model, dummy_inputs)
# net.load_state_dict(torch.load(trt_model_path))
predictor = create_mobilenetv1_ssd_predictor(model, candidate_size=200)
model.roi_head.box_predictor = predictor

def custom_loss(actual_labels, boxes, category, probs):

    print(f"Labels for this image: {actual_labels}")
    label_categories = torch.tensor([label["category_id"] for label in actual_labels])
    target_boxes = torch.tensor([label["bbox"] for label in actual_labels], dtype=float)

    print(f"Model output box tensor shape: {boxes.size()}")

    # Pad tensors to size
    if boxes.size()[0] > target_boxes.size()[0]:
        zeros = torch.zeros_like(boxes)
        if target_boxes.size()[0] == 0:
            pass
        else:
            zeros[:target_boxes.size()[0], :target_boxes.size()[1]] = target_boxes
        target_boxes = zeros
        #     target_boxes = F.pad(target_boxes, (boxes.size()[1],), value=0)

    else:
        zeros = torch.zeros_like(target_boxes)

        if boxes.size()[0] == 0:
            pass
        else:
            zeros[:boxes.size()[0], :boxes.size()[1]] = boxes
        boxes = zeros

    # Convert target boxes to (x1, y1, x2, y2) format
    target_boxes[:, 2:4] += target_boxes[:, 0:2]  # width + x1, height + y1
    target_boxes.requires_grad_()

    # Step 1: Match Predictions to Ground Truth
    iou_matrix = box_iou(boxes, target_boxes)  # Compute IoU between predictions and targets
    print(f"IOI Matrix: {iou_matrix}")
    matched_indices = torch.argmax(iou_matrix, dim=1)  # Match each prediction to the best ground truth
    print(f"Matched Indices: {matched_indices}")
    matched_gt = target_boxes[matched_indices]
    matched_labels = label_categories[matched_indices]

    # Step 2: False Positives and False Negatives

    unmatched_preds = iou_matrix.max(dim=1).values < 0.5  # Predictions with IoU < 0.5 are unmatched
    unmatched_targets = iou_matrix.max(dim=0).values < 0.5  # Ground truth boxes not matched by any prediction
    
    # unmatched = torch.where(iou_matrix < 0.5, iou_matrix, 0)
    # unmatched_pred_centroids = boxes[unmatched.argmax(dim=1)][0:1]
    # unmatched_target_centroids = boxes[unmatched.argmax(dim=0)][0:1]

    # Step 2.5: Centroid RMSE Loss
    # rmse_loss = torch.sqrt(torch.mean(torch.square(unmatched_pred_centroids - unmatched_target_centroids)))
    

    false_positive_loss = unmatched_preds.sum() * 0.1  # Penalize unmatched predictions
    false_negative_loss = unmatched_targets.sum() * 0.1  # Penalize unmatched ground truths

    # Step 3: Classification Loss (Cross Entropy)
    pred_probs = torch.nn.functional.softmax(category.float(), dim=-1)
    # print(f"Prediction Probability Shape: {pred_probs.size()}, Matched Labels Shape: {matched_labels.size()}")

    zeros = torch.zeros_like(matched_labels)
    zeros[:pred_probs.size()[0]] = pred_probs
    pred_probs = zeros

    target_probs = pred_probs.gather(dim=-1, index=matched_labels)

    zeros = torch.zeros_like(target_probs)
    zeros[:probs.size()[0]] = probs
    probs = zeros

    cls_loss = -torch.log(target_probs + 1e-6) * probs

    # Step 4: Regression Loss (GIoU Loss)
    reg_loss = generalized_box_iou_loss(boxes[~unmatched_preds], matched_gt[~unmatched_preds])

    # Combine Losses
    total_loss = cls_loss.mean() + reg_loss + false_positive_loss + false_negative_loss # + rmse_loss

    return total_loss.sum()

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = JSONDataset(transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# print(train_dataset.labels)

run = wandb.init(
    # Set the project where this run will be logged
    project="Terrawarden UAVVaste Training",
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
        labels = train_dataset.get_label(i)
        optimizer.zero_grad()
        # boxes, categories, probs = predictor.predict(images, 10, prob_threshold=0, no_grad=False)
        boxes, categories, probs = model(images)
        loss = custom_loss(labels, boxes, categories, probs)
        loss.backward()
        optimizer.step()

        wandb.log({"Step": epoch*len(train_loader) + i, "Loss": loss})
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# Save the trained model
torch.save(model.state_dict(), 'UAVVaste-Trained.pth')