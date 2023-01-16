import os
import json
import glob
import torch
import math
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from kmeans_pytorch import kmeans
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

import config as cfg

import pdb

class SatelliteDataset(Dataset):
    def __init__(self, data_paths, metadata=None, transform=None, device='cuda', shuffle=True, brightness=1.0) -> None:
        super().__init__()
        
        self.data_paths = data_paths
        self.device = device
        
        # Image enhancement
        self.brightness = brightness
        self.transform = transform

        if metadata:
            self.metadata = metadata
        else:
            self.metadata = self.extract_metadata(data_paths, shuffle=shuffle)
        print(f"Loaded a data set with {self.__len__()} images.")

    def extract_metadata(self, data_paths, shuffle=True):
        positive_data_paths = [os.path.join(data_path, "positive") for data_path in data_paths]
        negative_data_paths = [os.path.join(data_path, "negative") for data_path in data_paths]
        formats_list = ['.jpg', '.png']

        # Initialize the metadata list
        metadata = []
        
        for img_format in formats_list:
            for positive_data_path in positive_data_paths:
                # Record positive labels [0 stands for the positive labels]
                for img_path in glob.glob(positive_data_path + "/*" + img_format):
                    img_label = 0
                    metadata_ = {
                        "image_path": img_path,
                        "category_id": img_label,
                        "brightness": self.brightness
                    }
                    metadata.append(metadata_)
            for negative_data_path in negative_data_paths:
                # Record negative labels [1 stands for the negative labels]
                for img_path in glob.glob(negative_data_path + "/*" + img_format):
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
        if brightness != 1.0:
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
        """
        This function randomly selects a subset of the dataset which will be retained, given the fraction of the dataset.
        """
        assert fraction >= 0.0 and fraction <= 1.0
        total_pos, total_neg = self.get_posneg_count()
        keep_count = int(total_neg * fraction)
        positives, negatives = self.get_posneg()
        negatives = random.sample(negatives, keep_count)
        self.build_metadata_from_posneg(positives, negatives)
    
    def leave_number_of_negatives(self, number):
        """
        This function randomly selects a subset of the dataset which will be retained, given the number of
        images to be retained.
        """
        total_pos, total_neg = self.get_posneg_count()
        assert number >= 0 and number <= total_neg
        keep_count = number
        positives, negatives = self.get_posneg()
        negatives = random.sample(negatives, keep_count)
        self.build_metadata_from_posneg(positives, negatives)
    
    def remove_positives(self):
        _, negatives = self.get_posneg()
        self.build_metadata_from_posneg([], negatives)
    
    def remove_negatives(self):
        positives, _ = self.get_posneg()
        self.build_metadata_from_posneg(positives, [])
    
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
        return (positives.copy(), negatives.copy())
    
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
    
    def shuffle(self):
        random.shuffle(self.metadata)
    
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
        
    def KMeansAnalysis(self, K_max=5):
        def AverageKMeansError(cluster_centers, cluster_idxs, pixels, num_classes, device='cuda'):
            mean_error = torch.tensor(0, device=device, dtype=torch.float32)
            n_pixels = len(pixels)
            for k_idx in range(num_classes):
                active_pixels = (cluster_idxs == k_idx).float().to(device)
                mean_error += torch.sum(active_pixels * pixels.T)
            mean_error /= n_pixels
            return mean_error
        
        def SaveCenterColorsPlot(cluster_centers, save_dir):
            # Get the number of clusters
            n_centers = len(cluster_centers)
            
            # Get the number of rows and columns required
            n_rows = round(math.sqrt(n_centers))
            n_cols = math.ceil(n_centers / n_rows)
            
            # Plot the colors
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)

            for i in range(n_rows):
                for j in range(n_cols):
                    idx = i * n_cols + j
                    if idx >= n_centers:
                        break
                    axs[i][j].imshow([[cluster_centers[idx]]])
            fig.tight_layout()
            fig.savefig(save_dir)
            
        def SaveCenterColorsTensor(cluster_centers, save_dir):
            torch.save(cluster_centers, save_dir)
        
        def SaveErrorsPlot(k_errors, K_max, save_dir):
            plt.close()
            fig = plt.figure()
            plt.plot(range(2, K_max + 1), k_errors, 'b')
            plt.xlabel("# of clusters")
            plt.ylabel("Error")
            plt.title("Error vs # of clusters")
            plt.savefig(save_dir)
        
        assert K_max >= 2, "Number of clusters must be greater than 2."
        print("Running K-means analysis.")
        
        # Extract pixel values
        print("Extracting pixel values...")
        pixels = torch.empty(size=(0, 3), device=self.device)
        for idx in tqdm(range(len(self.metadata))):
            image, _ = self.__getitem__(idx)
            pixels_ = image.permute(1, 2, 0).view(-1, 3)
            pixels = torch.cat((pixels, pixels_))

        # Perform K-means clustering
        k_errors = []
        cluster_idxs_list = []
        cluster_centers_list = []
        for k in range(2, K_max + 1):
            print(f"K-means with {k} clusters.")
            km = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, tol=1e-04, random_state=0, verbose=0)
            cluster_idxs = km.fit_predict(pixels.cpu())
            cluster_centers = km.cluster_centers_
            mean_error = km.inertia_
            
            # Record the data
            k_errors.append(mean_error)
            cluster_idxs_list.append(cluster_idxs)
            cluster_centers_list.append(cluster_centers)

        # Extract the error gradients
        error_gradients = []
        for i in range(0, len(k_errors) - 1):
            change = k_errors[i + 1] - k_errors[i]
            error_gradients.append(-change)
        
        # Save cluster colors
        Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        for cluster_centers in cluster_centers_list:
            save_dir = os.path.join(cfg.RESULTS_DIR, f"cluster_centers_{len(cluster_centers)}.png")
            SaveCenterColorsPlot(cluster_centers, save_dir)
            save_dir = os.path.join(cfg.RESULTS_DIR, f"cluster_centers_{len(cluster_centers)}.pth")
            SaveCenterColorsTensor(cluster_centers, save_dir)
        
        # Save errors plot
        save_dir = os.path.join(cfg.RESULTS_DIR, "k_errors.png")
        SaveErrorsPlot(k_errors, K_max, save_dir)
        save_dir = os.path.join(cfg.RESULTS_DIR, "error_gradients.png")
        SaveErrorsPlot(error_gradients, K_max - 1, save_dir)