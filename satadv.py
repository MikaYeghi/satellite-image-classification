import os
import torch
from torch import nn
import torchvision.models as models
import glob
from pytorch3d.io import load_objs_as_meshes
from torch.nn import BCELoss
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.utils import save_image
from pathlib import Path
import numpy as np
import math
import seaborn as sn
import shutil
import torchvision.transforms.functional as F
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

from renderer import Renderer
from utils import random_unique_split, sample_random_elev_azimuth, create_model

import pdb

class SatAdv(nn.Module):
    def __init__(self, cfg, device='cuda:0'):
        super().__init__()
        
        self.device = device
        self.renderer = Renderer(device)
        self.cfg = cfg
        self.model = create_model(cfg, device)
        self.meshes = self.load_meshes()
        self.camouflages = self.load_camouflages()
        self.descriptive_colors = self.load_descriptive_colors()
        
        # Initialize parameters
        self.lights_direction = torch.tensor([0.0,-1.0,0.0], device=device, requires_grad=True).unsqueeze(0)
        self.distance = 5.0
        self.elevation = 90
        self.azimuth = -150
    
    def freeze_model(self):
        for name, param in self.named_parameters():
            if param.requires_grad and name.split('.')[0] == 'model':
                param.requires_grad = False
    
    def details(self):
        text = f""
        for name, param in self.named_parameters():
            text += f"{name}, {param.requires_grad}\n"
        return text
    
    def load_meshes(self, shuffle=True):
        print(f"Loading meshes from {self.cfg.MESHES_DIR}")
        meshes = []
        obj_paths = glob.glob(self.cfg.MESHES_DIR + "/*.obj")
        for obj_path in tqdm(obj_paths):
            mesh = load_objs_as_meshes([obj_path], device=self.device)[0]
            meshes.append(mesh)
        if shuffle:
            random.shuffle(meshes)
        return meshes
    
    def load_camouflages(self):
        """
        Loads camouflages from a given directory. The camouflages must be png files.
        """
        if self.cfg.DRESS_CAMOUFLAGE == 'fixed':
            glob_search = self.cfg.CAMOUFLAGE_TEXTURES_PATH + "/*.png"
        elif self.cfg.DRESS_CAMOUFLAGE == 'organic':
            glob_search = self.cfg.CAMOUFLAGE_TEXTURES_PATH + "/*/negative/*.png"
        camouflage_paths = glob.glob(glob_search) # Textures must be png files
        camouflages = []
        k = 0
        for camouflage_path in camouflage_paths:
            camouflage = Image.open(camouflage_path).convert("RGB")
            camouflage = pil_to_tensor(camouflage)[:3, ...]
            camouflage = camouflage / 255. if camouflage.dtype == torch.uint8 else camouflage
            camouflage = camouflage.to(self.device)
            camouflages.append(camouflage)
            # save_image(camouflage, f"results/cam_{k}.jpg")
            k += 1
        return camouflages
    
    def load_descriptive_colors(self):
        """
        Load a list of descriptive colors. If no path of descriptive colors is given, None is returned.
        If a path of descriptive colors is given, then this function loads a list of cluster centers (torch tensors). Each element of the list represents the color cluster centers coordinates for a different number of cluster centers.
        The lengths of elements of the returned array are like the following:
        [2, 3, 4, 5, ..., N]
        where each number represents the number of cluster centers encoded in each element for a total number of N-1 centers.
        """
        # Load the path of the descriptive colors
        descriptive_colors_path = self.cfg.DESCRIPTIVE_COLORS_PATH
        
        # No descriptive colors to load
        if descriptive_colors_path is None:
            return None
        
        cluster_centers_paths = glob.glob(descriptive_colors_path + "/*.pth")
        cluster_centers_list = []
        for cluster_centers_path in cluster_centers_paths:
            cluster_centers = torch.load(cluster_centers_path)
            cluster_centers_list.append(cluster_centers.clone())
        return cluster_centers_list
    
    def render_synthetic_image(self, mesh, background_image):
        return self.renderer.render(mesh, 
                                    background_image, 
                                     self.distance, 
                                     self.elevation, 
                                     self.azimuth, 
                                     self.lights_direction
                                    )
    
    def dress_camouflage(self, mesh, camouflage):
        mesh.textures._maps_padded.data = camouflage.unsqueeze(0).permute(0, 2, 3, 1).clone().to(self.device)
        return mesh
    
    def generate_pixelated_camouflage(self, block_size):
        assert 512 % block_size == 0, "512 is not divisible by the supplied block size."
        assert self.cfg.NUM_DESCRIPTIVE_COLORS >= self.cfg.COLORS_PER_CAMOUFLAGE, f"Not enough descriptive colors to generate {self.cfg.COLORS_PER_CAMOUFLAGE}-color camouflages."
        
        # Load cluster centers if they are available
        cluster_centers = None
        if self.descriptive_colors is None:
            raise ValueError("No descriptive colors loaded. Check descriptive colors loading.")
        else:
            for cluster_centers_ in self.descriptive_colors:
                if len(cluster_centers_) == self.cfg.NUM_DESCRIPTIVE_COLORS:
                    cluster_centers = cluster_centers_.clone()
            if cluster_centers is None:
                raise ValueError("Didn't find descriptive colors with the given number of clusters.")
        
        # Pick several random colors
        cluster_centers = cluster_centers[torch.randperm(cluster_centers.shape[0])] # random.shuffle creates problems (repeated colors), using this instead
        camouflage_colors = cluster_centers[:self.cfg.COLORS_PER_CAMOUFLAGE].to(self.device)
        
        # Generate the downsampled texture map
        downsampled_size = 512 // block_size # 512 is the texture map size
        camouflage = torch.empty((downsampled_size, downsampled_size, 3), device=self.device)
        H, W = camouflage.shape[:2]
        for h in range(H):
            for w in range(W):
                camouflage[h][w] = camouflage_colors[random.randint(0, self.cfg.COLORS_PER_CAMOUFLAGE - 1)]
        
        # Upscale
        transform = Resize(512, interpolation=F.InterpolationMode.NEAREST)
        camouflage = transform(camouflage.permute(2, 0, 1))
        # save_image(camouflage, "results/test.png")
        
        return camouflage
    
    def generate_random_organic_camouflage(self):
        # Number of clusters for each camouflage
        camouflage_n_centers = [4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4]
        
        # Select a random camouflage
        camouflage_idx = random.randint(1, 12)
        n_centers = camouflage_n_centers[camouflage_idx - 1]
        camouflage_path = os.path.join(self.cfg.CAMOUFLAGE_TEXTURES_PATH, f"Cam-{camouflage_idx}", "negative", f"Camouflage-{camouflage_idx}.png")
        camouflage = Image.open(camouflage_path).convert("RGB")
        camouflage = pil_to_tensor(camouflage)[:3, ...]
        camouflage = camouflage / 255. if camouflage.dtype == torch.uint8 else camouflage
        camouflage = camouflage.permute(1, 2, 0)
        camouflage = camouflage.to(self.device)
        
        # Load original cluster centers
        centers_path = os.path.join(self.cfg.CAMOUFLAGE_TEXTURES_PATH, f"Cam-{camouflage_idx}", "results", f"cluster_centers_{n_centers}.pth")
        cluster_centers = torch.from_numpy(torch.load(centers_path)).float().to(self.device)
        
        # Load new cluster centers
        new_centers_path = "/home/myeghiaz/Storage/descriptive-colors-real-train-fraction-0.1/cluster_centers_10.pth"
        new_cluster_centers = torch.load(new_centers_path).to(self.device)
        new_cluster_centers = new_cluster_centers[torch.randperm(new_cluster_centers.shape[0])]
        new_cluster_centers = new_cluster_centers[:n_centers]
        
        # Find the closest original cluster indices
        distances = torch.cdist(camouflage.float().view(-1, 3), cluster_centers.float().view(-1, 3))
        indices = torch.argmin(distances, dim=1).resize(512, 512)
        
        # Construct a new texture map
        new_camouflage = torch.empty((512, 512, 3))
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                new_camouflage[i][j] = new_cluster_centers[indices[i][j]]
        new_camouflage = new_camouflage.permute(2,0,1)
        
        # save_image(camouflage.permute(2, 0, 1), "results/original.png")
        # save_image(new_camouflage.permute(2, 0, 1), "results/new.png")
        
        return new_camouflage
    
    def generate_synthetic_subset(self, dataset, dataset_type, meshes, positive_limit=None, negative_limit=None):
        print(f"Generating {dataset_type} synthetic dataset.")
        positive_counter = 0
        negative_counter = 0
        negative_save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, dataset_type, "negative")
        positive_save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, dataset_type, "positive")
        
        # Get the total number of negative samples in the dataset
        total_positive, total_negative = dataset.get_posneg_count()
        
        # Remove all positive samples, as they are not used for synthetic dataset generation
        dataset.remove_positives()
        
        # Split the remaining samples into future positive and negative samples
        dataset.shuffle()
        if positive_limit is None and negative_limit is None:
            # Split equally
            pos_max_index = len(dataset) // 2
            neg_max_index = len(dataset)
        elif positive_limit is not None and negative_limit is None:
            assert positive_limit <= len(dataset), "Positive limit is greater than the total number of elements in the dataset."
            pos_max_index = positive_limit
            neg_max_index = len(dataset)
        elif positive_limit is None and negative_limit is not None:
            assert negative_limit <= len(dataset), "Negative limit is greater than the total number of elements in the dataset."
            pos_max_index = len(dataset) - negative_limit
            neg_max_index = len(dataset)
        elif positive_limit is not None and negative_limit is not None:
            assert positive_limit + negative_limit <= len(dataset), "Negative and positive limits sum is greater than the total number of elements in the dataset."
            pos_max_index = positive_limit
            neg_max_index = positive_limit + negative_limit
        else:
            raise NotImplementedError
        positive_files = dataset.get_posneg()[1][:pos_max_index].copy()
        negative_files = dataset.get_posneg()[1][pos_max_index:neg_max_index].copy()

        # Generate negative samples
        print(f"Generating {len(negative_files)} negative samples.")
        for negative_file in tqdm(negative_files):
            image_path = negative_file['image_path']
            save_path = os.path.join(negative_save_dir, f"image_{negative_counter}.png")
            shutil.copy(image_path, save_path)
            negative_counter += 1
        
        # Generate positive samples
        dataset.build_metadata_from_posneg(positive_files, [])
        print(f"Generating {len(positive_files)} positive samples.")
        for image, label in tqdm(dataset):
            mesh = random.choice(meshes)
            
            # Change textures to camouflage
            if self.cfg.DRESS_CAMOUFLAGE == 'fixed':
                mesh = self.dress_camouflage(mesh, random.choice(self.camouflages))
            elif self.cfg.DRESS_CAMOUFLAGE == 'random':
                random_camouflage = self.generate_pixelated_camouflage(block_size=self.cfg.PIXELATION_BLOCK_SIZE)
                mesh = self.dress_camouflage(mesh, random_camouflage)
            elif self.cfg.DRESS_CAMOUFLAGE == 'organic':
                organic_random_camouflage = self.generate_random_organic_camouflage()
                mesh = self.dress_camouflage(mesh, organic_random_camouflage)
            elif self.cfg.DRESS_CAMOUFLAGE is None:
                pass
            else:
                raise NotImplementedError
            # Positive class (i.e. with vehicle)
            # The numbers below were selected to make sure that the elevation is above 70 degrees
            distance = 5.0
            elevation, azimuth = sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0) 
            lights_direction = torch.tensor([random.uniform(-1, 1),-1.0,random.uniform(-1, 1)], device=self.device, requires_grad=True).unsqueeze(0)
            scaling_factor = random.uniform(0.70, 0.80)
            intensity = random.uniform(0.0, 1.0)
            
            # Render and save the image
            synthetic_image = self.renderer.render(
                mesh, 
                image, 
                elevation, 
                azimuth,
                lights_direction,
                scaling_factor=scaling_factor,
                intensity=intensity,
                ambient_color=((0.05, 0.05, 0.05),),
                distance=distance
            )
            save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, dataset_type, "positive", f"image_{positive_counter}.png")
            save_image(synthetic_image.permute(2, 0, 1), save_dir)
            positive_counter += 1
        print(f"Generated {positive_counter} positive images and {negative_counter} negative images.")
    
    def generate_synthetic_dataset(self, train_set, test_set):
        # Sample the meshes into training and testing meshes
        n_training_meshes = int(len(self.meshes) * self.cfg.TRAIN_MESHES_FRACTION)
        n_testing_meshes = len(self.meshes) - n_training_meshes
        train_meshes, test_meshes = random_unique_split(self.meshes, n_training_meshes, n_testing_meshes)
        
        # Check that the path exists. If not - create it
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "train", "positive")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "train", "negative")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "test", "positive")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "test", "negative")).mkdir(parents=True, exist_ok=True)
        
        self.generate_synthetic_subset(train_set, "train", train_meshes, positive_limit=self.cfg.POSITIVE_LIMIT_TRAIN, negative_limit=self.cfg.NEGATIVE_LIMIT_TRAIN)
        self.generate_synthetic_subset(test_set, "test", test_meshes, positive_limit=self.cfg.POSITIVE_LIMIT_TEST, negative_limit=self.cfg.NEGATIVE_LIMIT_TEST)
    
    def attack_image_mesh(self, mesh, background_image):
        lights_direction = torch.nn.Parameter(torch.tensor([0.0,-1.0,0.0], device=self.device, requires_grad=True).unsqueeze(0))
        intensity = torch.nn.Parameter(torch.tensor(1.0, device=self.device, requires_grad=True))
        image = self.renderer.render(mesh, background_image, lights_direction=lights_direction, elevation=self.elevation, azimuth=self.azimuth, intensity=intensity)
        
        # Save the original image
        plt.imshow(image.clone().detach().cpu().numpy())
        plt.savefig("results/test.png")
        plt.close('all')
        
        # Attack the image
        image = image.permute(2, 0, 1).unsqueeze(0)
        activation = nn.Sigmoid()
        with torch.no_grad():
            self.model.eval()
            preds = self.model(image)
            preds = activation(preds)
            preds = (preds > 0.5).float()
            # Save the original image
            plt.imshow(image[0].permute(1,2,0).clone().detach().cpu().numpy())
            plt.savefig("results/img_original.jpg")
            plt.close('all')
            print(preds)
        if preds.item() == 1:
            print("The model already predicts an incorrect class.")
            return
        else:
            # If the model still predicts a correct class, loop until the class is flipped
            self.model.train()
            reward_fn = BCELoss()
            # self.freeze_model()
            optimizer = torch.optim.Adam([lights_direction, intensity], lr=self.cfg.ATTACK_LR)
            correct_class = True
            labels_batched = torch.tensor([[0.0]], device=self.device)
            k = 0
            # Optimize
            while correct_class:
                activation = nn.Sigmoid()
                self.model.train()
                yhat = self.model(image)
                yhat = activation(yhat)
                loss = -reward_fn(yhat, labels_batched)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Generate the updated image
                image = self.renderer.render(mesh, background_image, lights_direction=lights_direction, elevation=self.elevation, azimuth=self.azimuth, intensity=intensity)

                if k % 100 == 0:
                    plt.imshow(image.clone().detach().cpu().numpy())
                    plt.savefig(f"results/img_{k}.jpg")
                    plt.close('all')
                k += 1

                # Evaluate
                self.model.eval()
                image = image.permute(2, 0, 1).unsqueeze(0)
                preds = self.model(image)
                preds = activation(preds)
                if yhat.item() > 0.5:
                    correct_class = False
                    print("Adversarial attack successful!")
                    plt.imshow(image[0].permute(1,2,0).clone().detach().cpu().numpy())
                    plt.savefig("results/img_final.jpg")
                    plt.close('all')
                
                print(f"Loss: {loss}. Train pred: {yhat}. Eval pred: {preds}\nLights direction: {lights_direction}\nIntensity: {intensity}\n")
    
    def get_lightdir_from_elaz(self, elev, azim):
        x = -math.cos(math.radians(elev)) * math.sin(math.radians(azim))
        y = -math.sin(math.radians(elev))
        z = -math.cos(math.radians(elev)) * math.cos(math.radians(azim))
        xyz = torch.tensor([x, y, z], device=self.device).unsqueeze(0)
        return xyz
    
    def find_failure_regions(self, mesh, background_image, elevs, azims, resolution=100, intensity=1.0):
        """
        elevs and azims represent lighting directions.
        """
        # Generate randomized parameters for this particular rendering (all except the tested one)
        elevation, azimuth = sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0) # Camera elevation and azimuth
        scaling_factor = random.uniform(0.70, 0.80)
        # elevation = 70
        # azimuth = 0
        # scaling_factor = 0.75
        
        # Plot an image sample
        # with torch.no_grad():
        #     plt.imshow(self.renderer.render(mesh, background_image, lights_direction=((0, -1, 0),), elevation=elevation, azimuth=azimuth, scaling_factor=scaling_factor, intensity=1.0).clone().detach().cpu().numpy())
        #     plt.savefig("results/image_sample.jpg")
        #     plt.close('all')
        
        # Create the activation function
        activation = nn.Sigmoid()
        
        # Correctness heatmap
        correctness_heatmap = torch.zeros((resolution, resolution))
        
        # Loop through all pixels in the correctness heatmap
        i = 0
        j = 0
        with torch.no_grad():
            for elev in tqdm(elevs):
                j = 0
                for azim in azims:
                    lights_direction = self.get_lightdir_from_elaz(elev, azim)
                    rendered_image = self.renderer.render(mesh, background_image, lights_direction=lights_direction, elevation=elevation, azimuth=azimuth, scaling_factor=scaling_factor, intensity=intensity)
                    rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0).float()

                    # Save the rendered image
                    if self.cfg.VISUALIZE_HEATMAP_SAMPLES:
                        save_image(rendered_image[0], f"results/image_{i}_{j}.jpg")

                    # Run inference on the image
                    self.model.eval()
                    prediction = activation(self.model(rendered_image)).item()

                    # Update the heatmap
                    heatmap_pixel = 1 - prediction # Correct class is 0, hence invert
                    correctness_heatmap[i][j] = heatmap_pixel

                    j += 1
                i += 1
        
        # # Plot the heatmap
        # plt.imshow(correctness_heatmap, cmap='Blues')
        # plt.savefig("results/test.jpg")
        # plt.close('all')
        
        return correctness_heatmap
    
    def failure_analysis(self, data_set, resolution=100, n_samples=100, plot=True, intensity=1.0):
        # Initialize the dataset correctness tensor
        dataset_correctness = torch.empty(size=(0, resolution, resolution), device=self.device)
        
        # Find the angle ranges
        elevs = np.linspace(0, 90, resolution)
        azims = np.linspace(-180, 180, resolution)
        
        # Loop through the dataset
        samples_count = 0
        for image, label in data_set:
            # If the number of required samples has been reached
            if samples_count >= n_samples:
                break
                
            if label == 1: # take only empty images
                print(f"Sample: {samples_count + 1}")
                mesh = random.choice(self.meshes)
                correctness_image = self.find_failure_regions(mesh, image, elevs, azims, resolution=resolution, intensity=intensity)
                correctness_image = correctness_image.to(self.device)
                dataset_correctness = torch.cat((dataset_correctness, correctness_image.unsqueeze(0)))
                samples_count += 1
            else:
                pass

        average_correctness = dataset_correctness.mean(dim=0)
        std_correctness = dataset_correctness.std(dim=0)
        
        # Save the heatmaps tensor (all heatmaps)
        torch.save(dataset_correctness, os.path.join(self.cfg.RESULTS_DIR, f"tensor_{self.cfg.HEATMAP_NAME}.pt"))
        
        # Plot the heatmap
        if plot:
            plt.close('all')
            plt.figure(figsize=(12.8, 9.6))
            heatmap = sn.heatmap(average_correctness.cpu(), xticklabels=azims.astype(np.int), yticklabels=elevs.astype(np.int))
            plt.xlabel("Azimuth")
            plt.ylabel("Elevation")
            plt.title(f"Average probability for different light directions with intensity level {intensity}")
            plt.savefig(os.path.join(self.cfg.RESULTS_DIR, f"mean_{self.cfg.HEATMAP_NAME}.jpg"))
            plt.close('all')
            plt.figure(figsize=(12.8, 9.6))
            heatmap = sn.heatmap(std_correctness.cpu(), xticklabels=azims.astype(np.int), yticklabels=elevs.astype(np.int))
            plt.xlabel("Azimuth")
            plt.ylabel("Elevation")
            plt.title("Standard deviation of data correctness")
            plt.savefig(os.path.join(self.cfg.RESULTS_DIR, f"std_{self.cfg.HEATMAP_NAME}.jpg"))
            plt.close('all')
        
        return average_correctness