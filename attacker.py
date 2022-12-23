import os
import torch
import torchvision
from tqdm import tqdm
from torch import nn
from torchvision.utils import save_image
from pathlib import Path

from renderer import Renderer
from losses import BCELoss, ColorForce, BCEColor

import pdb

class FGSMAttacker:
    def __init__(self, model, attacked_params, save_dir, epsilon=1e-3, device='cuda'):
        self.model = model
        self.attacked_params = attacked_params
        self.epsilon = epsilon
        self.device = device
        self.renderer = Renderer(device=device)
        
        # Create the save directories
        self.save_dir = save_dir
        Path(os.path.join(save_dir, "original")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_dir, "adversarial")).mkdir(parents=True, exist_ok=True)
        
        # torch.Tensor to PIL.Image converter
        self.converter = torchvision.transforms.ToPILImage()
        
        # List which stored AttackedImage variables
        self.adversarial_examples_list = []
        
        # Set up the model
        self.activation = nn.Sigmoid() # Need this to apply to the model output
        self.loss_fn = BCELoss()
    
    def attack_single_image(self, attacked_image, true_label=0):
        # Set the model to eval mode following the PyTorch tutorial for adversarial attacks
        self.model.eval()
        self.model.zero_grad()
        
        # Set up parameters which require gradients for this attack
        rendering_params = attacked_image.get_rendering_params()
        rendering_params = self.select_attacked_params(rendering_params, self.attacked_params)
        
        # Attack the image
        k = 0
        while True:
            # Render the image
            image = self.renderer.render(
                mesh=rendering_params['mesh'],
                background_image=rendering_params['background_image'],
                elevation=rendering_params['elevation'],
                azimuth=rendering_params['azimuth'],
                lights_direction=rendering_params['lights_direction'],
                distance=rendering_params['distance'],
                scaling_factor=rendering_params['scaling_factor'],
                intensity=rendering_params['intensity'],
                ambient_color=rendering_params['ambient_color']
            ).permute(2, 0, 1).unsqueeze(0)
            if k == 0:
                attacked_image.set_original_image(image[0])
            
            # Run prediction on the image
            pred = self.activation(self.model(image))
            print(pred.item())
            
            # If the class is incorrect, then stop. Otherwise, perform a single attack step
            if (pred > 0.5).int().item() != true_label:
                attacked_image.set_adversarial_image(image[0])
                attacked_image.set_adversarial_params(rendering_params)
                self.adversarial_examples_list.append(attacked_image)
                break
            else:
                # Calculate the loss
                label_batched = torch.tensor([true_label], device=self.device, dtype=torch.float).unsqueeze(0)
                loss = self.loss_fn(pred, label_batched)

                # Propagate the gradients
                loss.backward()
            
                # Call FGSM attack
                rendering_params = self.fgsm_attack(rendering_params)
                
                # Zero all gradients
                self.zero_gradients(rendering_params)
            k += 1
            
    def fgsm_attack(self, rendering_params):
        for param in self.attacked_params:
            if param == 'textures':
                # Collect the element-wise sign of the data gradient
                sign_grad = rendering_params['mesh'].textures.maps_padded().grad.data.sign()
                
                # Perturb the data
                rendering_params['mesh'].textures.maps_padded().data = rendering_params['mesh'].textures.maps_padded().data + self.epsilon * sign_grad
            elif attacked_param == 'mesh':
                # TODO: make everything related to the mesh (textures, vertices) differentiable
                raise NotImplementedError
            elif param in rendering_params:
                # Collect the element-wise sign of the data gradient
                sign_grad = rendering_params[param].grad.data.sign()
                
                # Perturb the data
                rendering_params[param].data = rendering_params[param].data + self.epsilon * sign_grad
        
        return rendering_params
    
    def select_attacked_params(self, rendering_params, attacked_params):
        for attacked_param in attacked_params:
            if attacked_param == 'textures':
                for i in range(len(rendering_params['mesh'].textures.maps_list())):
                    rendering_params['mesh'].textures.maps_padded().requires_grad = True
                    # rendering_params['mesh'].textures.maps_padded().retain_grad()
                pass
            elif attacked_param == 'mesh':
                # TODO: make everything related to the mesh (textures, vertices) differentiable
                raise NotImplementedError
            elif attacked_param in rendering_params:
                rendering_params[attacked_param].requires_grad = True

        return rendering_params
    
    def zero_gradients(self, rendering_params):
        self.model.zero_grad()
        
        for param in self.attacked_params:
            if param == 'textures':
                rendering_params['mesh'].textures.maps_padded().grad.zero_()
            elif param == 'mesh':
                raise NotImplementedError
            elif param in self.rendering_params:
                rendering_params[param].grad.zero_()
    
    def __str__(self):
        text = f"Epsilon: {self.epsilon}.\nDevice: {self.device}.\nAttacked parameters: {self.attacked_params}.\nNumber of correct-adversarial pairs: {len(self.adversarial_examples_list)}."
        return text
    
    def save(self):
        print(f"Saving correct-adversarial image pairs to {self.save_dir}")
        idx = 0
        for attacked_image in self.adversarial_examples_list:
            # Retrieve the original and final images
            original_image = attacked_image.get_original_image()
            adversarial_image = attacked_image.get_adversarial_image()
            
            # Save each image to the relevant folder
            original_save_path = os.path.join(self.save_dir, "original", f"image_{idx}.jpg")
            adversarial_save_path = os.path.join(self.save_dir, "adversarial", f"image_{idx}.jpg")
            
            # Convert the images to PIL
            original_image = self.converter(original_image)
            adversarial_image = self.converter(adversarial_image)
            
            # Save the images with high quality
            original_image.save(original_save_path, quality=95)
            adversarial_image.save(adversarial_save_path, quality=95)
            
            idx += 1
            
    def get_num_pairs(self):
        return len(self.adversarial_examples_list)