import torch
from torch.utils.data import Dataset, DataLoader
import json 
import os
import PIL

base_path = "/home/krsiegall/Terrawarden/UAVVaste"

class JSONDataset(Dataset):
    def __init__(self, 
                image_root = os.path.join(base_path,"images"), 
                json_filename = os.path.join(base_path,"annotations/annotations.json"), 
                transforms = None):
        self.image_root = image_root
        self.labels = {}
        self.transformations = transforms
        self.process_annotations(json_filename)
        print(f"JSON Loaded successfully, with {len(self.json["images"])} images and {len(self.json["annotations"])} annotations.")

    def __len__(self):
        return len(self.json["images"])

    def __getitem__(self, idx):
        img_name = self.json["images"][idx]["file_name"]
        img_path = os.path.join(self.image_root, img_name)
        img = PIL.Image.open(img_path).convert("RGB")
        img = self.transformations(img)
        print(f"Retrieved image {idx}, with a size of {img.size()}")
        return img, img

    def get_label(self, idx):
        print(f"Image {idx} has {len(self.labels[idx])} labels")
        return [self.json["annotations"][id] for id in self.labels[idx]]

    def process_annotations(self, filepath):
        print(filepath)
        with open(filepath, "r") as json_file:
            self.json = json.load(json_file)

        for i in range(len(self.json["images"])):
            for detected_item in self.json["annotations"]:
                if i == detected_item["image_id"]:
                    if i in self.labels:
                        self.labels[i].append(detected_item["id"])
                    else:
                        self.labels[i] = []
                        self.labels[i].append(detected_item["id"])