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

from renderer import Renderer
from utils import random_unique_split

import pdb

class SatAdv(nn.Module):
    def __init__(self, cfg, device='cuda:0'):
        super().__init__()
        
        self.device = device
        self.renderer = Renderer(device)
        self.cfg = cfg
        self.model = self.create_model()
        self.meshes = self.load_meshes()
        
        # Initialize parameters
        self.lights_direction = torch.nn.Parameter(torch.tensor([0.0,1.0,0.0], device=device, requires_grad=True).unsqueeze(0))
        self.distance = 5.0
        self.elevation = 90
        self.azimuth = -150
    
    def create_model(self):
        model = models.resnet101(pretrained=True)
        model.fc = torch.nn.Linear(2048, 1, device=self.device, dtype=torch.float32)
        if self.cfg.MODEL_WEIGHTS:
            print(f"Loading model weights from {self.cfg.MODEL_WEIGHTS}")
            model.load_state_dict(torch.load(self.cfg.MODEL_WEIGHTS))
        model.to(self.device)
        
        return model
    
    def freeze_model(self):
        for name, param in self.named_parameters():
            if param.requires_grad and name.split('.')[0] == 'model':
                param.requires_grad = False
    
    def details(self):
        text = f""
        for name, param in self.named_parameters():
            text += f"{name}, {param.requires_grad}\n"
        return text
    
    def load_meshes(self):
        print(f"Loading meshes from {self.cfg.MESHES_DIR}")
        meshes = []
        obj_paths = glob.glob(self.cfg.MESHES_DIR + "/*.obj")
        for obj_path in obj_paths:
            mesh = load_objs_as_meshes([obj_path], device=self.device)[0]
            meshes.append(mesh)
        return meshes
    
    def render_synthetic_image(self, mesh, background_image):
        return self.renderer.render(mesh, 
                                    background_image, 
                                     self.distance, 
                                     self.elevation, 
                                     self.azimuth, 
                                     self.lights_direction
                                    )
    
    def generate_synthetic_dataset(self, train_set, test_set):
        # Sample the meshes into training and testing meshes
        n_training_meshes = len(self.meshes) // 2
        n_testing_meshes = len(self.meshes) - n_training_meshes
        train_meshes, test_meshes = random_unique_split(self.meshes, n_training_meshes, n_testing_meshes)
        
        # Train set
        print("Generating training synthetic dataset.")
        positive_counter = 0
        negative_counter = 0
        for image, label in tqdm(train_set):
            if label == 1: # select only negative samples, i.e. without real cars
                for mesh in train_meshes: # place each vehicle in the image
                    random_number = random.uniform(0, 1)
                    if random_number > 0.5:
                        # Negative class (i.e. background)
                        synthetic_image = image
                        save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "train", "negative", f"image_{negative_counter}.png")
                        save_image(synthetic_image, save_dir)
                        negative_counter += 1
                    else:
                        # Positive class (i.e. with vehicle)
                        
                        # Generate randomized parameters for rendering
                        distance = random.uniform(4.8, 5.2)
                        elevation = random.uniform(70, 110)
                        azimuth = random.uniform(0, 360)
                        lights_direction = self.lights_direction.clone()
                        scaling_factor = random.uniform(0.80, 0.90)
                        
                        # Render and save the image
                        synthetic_image = self.renderer.render(
                            mesh, 
                            image,
                            distance, 
                            elevation, 
                            azimuth,
                            lights_direction,
                            scaling_factor=scaling_factor,
                        )
                        save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "train", "positive", f"image_{positive_counter}.png")
                        save_image(synthetic_image.permute(2, 0, 1), save_dir)
                        positive_counter += 1
            if positive_counter >= 10000:
                break

        print(f"Generated {positive_counter} positive images and {negative_counter} negative images for the training set.")
        
        # Train set
        print("Generating testing synthetic dataset.")
        positive_counter = 0
        negative_counter = 0
        for image, label in tqdm(test_set):
            if label == 1: # select only negative samples, i.e. without real cars
                for mesh in test_meshes: # place each vehicle in the image
                    random_number = random.uniform(0, 1)
                    if random_number > 0.5:
                        # Negative class (i.e. background)
                        synthetic_image = image
                        save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "test", "negative", f"image_{negative_counter}.png")
                        save_image(synthetic_image, save_dir)
                        negative_counter += 1
                    else:
                        # Positive class (i.e. with vehicle)
                        
                        # Generate randomized parameters for rendering
                        distance = random.uniform(4.8, 5.2)
                        elevation = random.uniform(70, 110)
                        azimuth = random.uniform(0, 360)
                        lights_direction = self.lights_direction.clone()
                        scaling_factor = random.uniform(0.55, 0.65)
                        
                        # Render and save the image
                        synthetic_image = self.renderer.render(
                            mesh, 
                            image,
                            distance, 
                            elevation, 
                            azimuth,
                            lights_direction,
                            scaling_factor=scaling_factor,
                        )
                        save_dir = os.path.join(self.cfg.SYNTHETIC_SAVE_DIR, "test", "positive", f"image_{positive_counter}.png")
                        save_image(synthetic_image.permute(2, 0, 1), save_dir)
                        positive_counter += 1
            if positive_counter >= 1000:
                break
            
        print(f"Generated {positive_counter} positive images and {negative_counter} negative images for the testing set.")
    
    def attack_image_mesh(self, mesh, background_image):
        image = self.render_synthetic_image(mesh, background_image)
        plt.imshow(image.clone().detach().cpu().numpy())
        plt.savefig("results/test.png")
        image = image.permute(2, 0, 1).unsqueeze(0)
        activation = nn.Sigmoid()
        with torch.no_grad():
            self.model.eval()
            preds = self.model(image)
            preds = activation(preds)
            preds = (preds > 0.5).float()
            print(preds)
        if preds.item() == 1:
            # If the model already predicts an incorrect class
            print("The model already predicts an incorrect class.")
            return
        else:
            # If the model still predicts a correct class, loop until the class is flipped
            # Initial setup
            self.model.train()
            reward_fn = BCELoss()
            self.freeze_model()
            optimizer = torch.optim.Adam([self.lights_direction], lr=self.cfg.LR)
            activation = nn.Sigmoid()
            correct_class = True
            labels_batched = torch.tensor([[0.0]], device=self.device)
            k = 0
            # Optimize
            while correct_class:
                yhat = self.model(image)
                yhat = activation(yhat)
                loss = -reward_fn(yhat, labels_batched)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                image = self.render_synthetic_image(mesh, background_image).permute(2, 0, 1).unsqueeze(0)

                print(f"Loss: {loss}. Prediction: {yhat}.\nLights direction: {self.lights_direction}\n")

                if k % 100 == 0:
                    plt.imshow(image[0].permute(1,2,0).clone().detach().cpu().numpy())
                    plt.savefig(f"results/img_{k}.jpg")
                    plt.close('all')
                k += 1