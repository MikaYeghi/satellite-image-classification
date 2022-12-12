import os
import json
import glob
from tqdm import tqdm
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import random

import pdb

class SatelliteDataset(Dataset):
    def __init__(self, data_path, metadata=None, transform=None, device='cuda', shuffle=True, brightness=1.0) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.device = device
        
        # Image enhancement
        self.brightness = brightness
        self.transform = transform

        if metadata:
            self.metadata = metadata
        else:
            self.metadata = self.extract_metadata(data_path, shuffle=shuffle)
        print(f"Loaded a data set with {self.__len__()} images.")

    def extract_metadata(self, data_path, shuffle=True):
        positive_data_path = os.path.join(data_path, "positive")
        negative_data_path = os.path.join(data_path, "negative")
        
        # Initialize the metadata list
        metadata = []
        
        # Record positive labels [0 stands for the positive labels]
        for img_path in glob.glob(positive_data_path + "/*.png"):
            img_label = 0
            metadata_ = {
                "image_path": img_path,
                "category_id": img_label,
                "brightness": self.brightness
            }
            metadata.append(metadata_)
        
        # Record negative labels [1 stands for the negative labels]
        for img_path in glob.glob(negative_data_path + "/*.png"):
            img_label = 1
            metadata_ = {
                "image_path": img_path,
                "category_id": img_label,
                "brightness": self.brightness
            }
            metadata.append(metadata_)
        if shuffle:
            random.shuffle(metadata)
        return metadata
    
    def __getitem__(self, idx):
        data = self.metadata[idx]
        
        # Extract the label
        label = data['category_id']
        
        # Extract the image
        image = Image.open(data['image_path']).convert('RGB') # force RGB instead of RGBA
        
        # Extract image brightness
        brightness = data['brightness']
        
        # Modify image brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        if self.transform:
            image = self.transform(image).to(self.device)
        else:
            image = pil_to_tensor(image).to(self.device)
        
        # Remove the 4th channel
        image = image[:3, ...]
        
        return (image, label)
    
    def __len__(self):
        return len(self.metadata)
    
    def get_metadata(self):
        return self.metadata
    
    def set_brightness(self, brightness):
        self.brightness = brightness
        for idx in range(len(self.metadata)):
            self.metadata[idx]['brightness'] = brightness
        
    def leave_fraction_of_negatives(self, fraction):
        assert fraction >= 0.0 and fraction <= 1.0
        total_pos, total_neg = self.get_posneg_count()
        keep_count = int(total_neg * fraction)
        positives, negatives = self.get_posneg()
        negatives = random.sample(negatives, keep_count)
        self.build_metadata_from_posneg(positives, negatives)
    
    def build_metadata_from_posneg(self, positives, negatives):
        metadata = positives + negatives
        random.shuffle(metadata)
        self.metadata = metadata
    
    def get_posneg(self):
        positives = []
        negatives = []
        for data in self.metadata:
            if data['category_id'] == 0:
                positives.append(data)
            elif data['category_id'] == 1:
                negatives.append(data)
            else:
                raise NotImplementedError
        return (positives, negatives)
    
    def get_posneg_count(self):
        total_pos = 0
        total_neg = 0
        for data in self.metadata:
            if data['category_id'] == 0:
                total_pos += 1
            elif data['category_id'] == 1:
                total_neg += 1
            else:
                raise NotImplementedError
        
        return (total_pos, total_neg)
    
    def details(self):
        total_pos, total_neg = self.get_posneg_count()
        text = f"Positive: {total_pos}. Negative: {total_neg}."
        return text
    
    def augment_brightness(self, brightness_levels):
        brightness_levels = [x for x in brightness_levels if x != 0.0] # exclude 0.0 to avoid confusion during training
        print("Augmenting the data set.")
        new_metadata = []
        for data in tqdm(self.metadata):
            for brightness in brightness_levels:
                data_ = data.copy()
                data_['brightness'] = brightness
                new_metadata.append(data_)
        self.metadata = new_metadata