from torchvision import transforms

import config as cfg

class SatTransforms:
    def __init__(self):
        if cfg.APPLY_TRAIN_TRANSFORMS:
            self.train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.3),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomRotation(degrees=180)
            ])
        else:
            self.train_transforms = transforms.Compose([transforms.ToTensor()])
        self.test_transforms = transforms.Compose([transforms.ToTensor()])
    
    def get_train_transforms(self):
        return self.train_transforms
    
    def get_test_transforms(self):
        return self.test_transforms