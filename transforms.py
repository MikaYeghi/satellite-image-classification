from torchvision import transforms

class SatTransforms:
    def __init__(self):
        self.train_transforms = transforms.Compose([transforms.ToTensor()])
        self.test_transforms = transforms.Compose([transforms.ToTensor()])
    
    def get_train_transforms(self):
        return self.train_transforms
    
    def get_test_transforms(self):
        return self.test_transforms