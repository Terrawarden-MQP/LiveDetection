import torch
from torch.utils.data import Dataset, DataLoader

class JSONDataset(Dataset):
    def __init__(self, 
                image_root = "../UAVVaste/images", 
                json_filename = "../UAVVaste/annotations/annotations.json", 
                transforms = None):
        self.json = self.process_annotations(filename)
        self.transformations = transforms
        self.labels = {}

    def __len__(self):
        return len(self.json["images"])

    def __getitem__(self, idx):
        img_name = self.json["images"][idx]
        img = PIL.Image.open(img_path).convert("RGB")
        img = self.transformations(img)
        return img, self.labels[idx]


    def process_annotations(self, filepath):
        self.json = json.loads(filepath)

        for i in range(len(self.json["images"])):
            for detected_item in self.json["annotations"]:
                if i == detected_item["image_id"]:
                    if i in self.labels:
                        self.labels[i] += detected_item["id"]
                    else:
                        self.labels[i] = [detected_item["id"]]