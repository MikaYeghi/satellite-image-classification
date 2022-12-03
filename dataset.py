import os
import json
import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

import pdb

class SatelliteDataset(Dataset):
    def __init__(self, data_path, transform=None, device='cuda') -> None:
        super().__init__()
        
        self.data_path = data_path
        self.transform = transform
        self.device = device
        
        self.metadata = self.extract_metadata(data_path)
        print(f"Loaded a data set with {self.__len__()} images.")

    def extract_metadata(self, data_path):
        positive_data_path = os.path.join(data_path, "positive")
        negative_data_path = os.path.join(data_path, "negative")
        
        # Initialize the metadata list
        metadata = []
        
        # Record positive labels [0 stands for the positive labels]
        for img_path in glob.glob(positive_data_path + "/*.png"):
            img_label = 0
            metadata_ = {
                "image_path": img_path,
                "category_id": img_label
            }
            metadata.append(metadata_)
        
        # Record negative labels [1 stands for the negative labels]
        for img_path in glob.glob(negative_data_path + "/*.png"):
            img_label = 1
            metadata_ = {
                "image_path": img_path,
                "category_id": img_label
            }
            metadata.append(metadata_)
        
        return metadata
    
    def __getitem__(self, idx):
        data = self.metadata[idx]
        
        # Extract the label
        label = data['category_id']
        
        # Extract the image
        # NOTE: consider removing the 4-th channel
        image = Image.open(data['image_path'])
        if self.transform:
            image = self.transform(image).to(self.device)
        else:
            image = pil_to_tensor(image).to(self.device)
        
        # Remove the 4th channel
        image = image[:3, ...]
        
        return (image, label)
    
    def __len__(self):
        return len(self.metadata)