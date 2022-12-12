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
        for obj_path in obj_paths:
            mesh = load_objs_as_meshes([obj_path], device=self.device)[0]
            meshes.append(mesh)
        if shuffle:
            random.shuffle(meshes)
        return meshes
    
    def render_synthetic_image(self, mesh, background_image):
        return self.renderer.render(mesh, 
                                    background_image, 
                                     self.distance, 
                                     self.elevation, 
                                     self.azimuth, 
                                     self.lights_direction
                                    )
    
    def generate_synthetic_subset(self, dataset, dataset_type, meshes):
        print(f"Generating {dataset_type} synthetic dataset.")
        positive_counter = 0
        negative_counter = 0
        for image, label in tqdm(dataset):
            if label == 1: # select only negative samples, i.e. without real cars
                for mesh in meshes: # place each vehicle in the image
                    random_number = random.uniform(0, 1)
                    if random_number > 0.5:
                        # Negative class (i.e. background)
                        synthetic_image = image
                        save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, dataset_type, "negative", f"image_{negative_counter}.png")
                        save_image(synthetic_image, save_dir)
                        negative_counter += 1
                    else:
                        # Positive class (i.e. with vehicle)
                        
                        # Generate randomized parameters for rendering
                        distance = 5.0
                        elevation, azimuth = sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0) # The numbers were selected to make sure that the elevation is above 70 degrees
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
            if positive_counter >= 1000:
                break
        print(f"Generated {positive_counter} positive images and {negative_counter} negative images for the training set.")
    
    def generate_synthetic_dataset(self, train_set, test_set):
        # Sample the meshes into training and testing meshes
        n_training_meshes = len(self.meshes) // 2
        n_testing_meshes = len(self.meshes) - n_training_meshes
        train_meshes, test_meshes = random_unique_split(self.meshes, n_training_meshes, n_testing_meshes)
        
        # Check that the path exists. If not - create it
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "train", "positive")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "train", "negative")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "test", "positive")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "test", "negative")).mkdir(parents=True, exist_ok=True)
        
        self.generate_synthetic_subset(train_set, "train", train_meshes)
        self.generate_synthetic_subset(test_set, "test", test_meshes)
    
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
    
    def find_failure_regions(self, mesh, background_image, resolution=100):
        # Generate randomized parameters for this particular rendering (all except the tested one)
        elevation, azimuth = sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0)
        scaling_factor = random.uniform(0.70, 0.80)
        
        # Plot an image sample
        with torch.no_grad():
            plt.imshow(self.renderer.render(mesh, background_image, lights_direction=((0, -1, 0),), elevation=elevation, azimuth=azimuth, scaling_factor=scaling_factor, intensity=1.0).clone().detach().cpu().numpy())
            plt.savefig("results/image_sample.jpg")
            plt.close('all')
        
        # Create the activation function
        activation = nn.Sigmoid()
        
        # Generate the range of light directions that need to be tested
        light_directions_per_axis = np.linspace(-1, 1, num=resolution)
        
        # Correctness heatmap
        correctness_heatmap = torch.zeros((resolution, resolution))
        
        # Loop through all pixels in the correctness heatmap
        i = 0
        j = 0
        with torch.no_grad():
            for lights_direction_x in tqdm(light_directions_per_axis):
                j = 0
                for lights_direction_z in light_directions_per_axis:
                    lights_direction = torch.tensor([lights_direction_x, -1, lights_direction_z], device=self.device).unsqueeze(0)
                    rendered_image = self.renderer.render(mesh, background_image, lights_direction=lights_direction, elevation=elevation, azimuth=azimuth, scaling_factor=scaling_factor, intensity=1.0)
                    rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0).float()

                    # Run inference on the image
                    self.model.eval()
                    prediction = activation(self.model(rendered_image)).item()
                    
                    # Update the heatmap
                    heatmap_pixel = 1 - prediction # Correct class is 0, hence invert
                    correctness_heatmap[i][j] = heatmap_pixel
                    
                    j += 1
                i += 1
        
        # Plot the heatmap
        plt.imshow(correctness_heatmap, cmap='Blues')
        plt.savefig("results/test.jpg")
        plt.close('all')
        
        return correctness_heatmap