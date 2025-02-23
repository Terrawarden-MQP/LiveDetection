import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
import transforms as CustomT
from coco_utils import get_coco
import utils
from engine import train_one_epoch, evaluate

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_transform(train):
    transforms = []

    transforms.append(T.ToTensor())

    if train:
        transforms.append(CustomT.RandomHorizontalFlip(0.5))
        transforms.append(CustomT.RandomZoomOut())
        transforms.append(CustomT.RandomPhotometricDistort())
        transforms.append(CustomT.RandomIoUCrop())
        transforms.append(T.GaussianBlur(kernel_size=(5,5)))
        transforms.append(T.GaussianNoise())

    transforms.append(T.ToDtype(torch.float, scale=True))
    
    return T.Compose(transforms)

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

train_dataset = get_coco(root = "../../../TACO/", 
                        image_set="train", 
                        transforms=get_transform(train=True),
                        with_masks=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

test_dataset = get_coco(root = "../../../TACO/", 
                        image_set="test", 
                        transforms=get_transform(train=False),
                        with_masks=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

val_dataset = get_coco(root = "../../../TACO/", 
                        image_set="val", 
                        transforms=get_transform(train=False),
                        with_masks=True)
val_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 60
# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    params,
    lr=0.005,
    # momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 10

import wandb
run = wandb.init(
    # Set the project where this run will be logged
    project="Terrawarden TACO Training",
    # Track hyperparameters and run metadata
    config={
        "num_classes": 60,
        "steps_per_epoch": len(train_loader),
        "epochs": num_epochs,
    },
)


for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10, wandb=True)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    test_evaluator = evaluate(model, test_loader, device=device)
    wandb.log(test_evaluator.coco_eval)
val_evaluator = evaluate(model, val_loader, device=device)
print(val_evaluator.coco_eval)
print("Training Complete!")

torch.save(model.state_dict(), 'TACO-Trained.pth')