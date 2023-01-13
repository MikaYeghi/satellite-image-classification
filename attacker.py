import os
import torch
import torchvision
from tqdm import tqdm
from torch import nn
from torchvision.utils import save_image
from pathlib import Path

from renderer import Renderer
from losses import BCELoss, ColorForce, BCEColor
from utils import blend_images

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
        Path(os.path.join(save_dir, "positive")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_dir, "negative")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_dir, "difference")).mkdir(parents=True, exist_ok=True)
        
        # torch.Tensor to PIL.Image converter
        self.converter = torchvision.transforms.ToPILImage()
        
        # List which stores AttackedImage variables
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
            # image = torch.clip(image, min=0.0, max=1.0)
            
            if k == 0:
                attacked_image.set_original_image(image[0].clone())
                
            # Check for shadow attacks
            if 'shadow_image' in rendering_params:
                image = blend_images(image, rendering_params['shadow_image'])
            
            # Run prediction on the image
            pred = self.activation(self.model(image))
            
            # If the class is incorrect, then stop. Otherwise, perform a single attack step
            if (pred > 0.5).int().item() != true_label:
                attacked_image.set_adversarial_image(image[0].clone())
                attacked_image.set_rendering_params(rendering_params)
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
            elif param == 'mesh':
                # TODO: make everything related to the mesh (textures, vertices) differentiable
                raise NotImplementedError
            elif param in rendering_params:
                sign_grad = rendering_params[param].grad.data.sign()
                rendering_params[param].data = rendering_params[param].data + self.epsilon * sign_grad
            elif param == 'shadows':
                sign_grad = rendering_params['shadow_image'].grad.data.sign()
                rendering_params['shadow_image'].data = rendering_params['shadow_image'].data + self.epsilon * sign_grad
        
        return rendering_params
    
    def select_attacked_params(self, rendering_params, attacked_params):
        for attacked_param in attacked_params:
            if attacked_param == 'textures':
                for i in range(len(rendering_params['mesh'].textures.maps_list())):
                    rendering_params['mesh'].textures.maps_padded().requires_grad = True
            elif attacked_param == 'mesh':
                # TODO: make everything related to the mesh (textures, vertices) differentiable
                raise NotImplementedError
            elif attacked_param in rendering_params:
                rendering_params[attacked_param].requires_grad = True
            elif attacked_param == 'shadows':
                H, W = rendering_params['background_image'].shape[1:3]
                shadow_image = torch.rand(size=(1, H, W), device=self.device)
                rendering_params['shadow_image'] = shadow_image.clone()
                rendering_params['shadow_image'].requires_grad = True

        return rendering_params
    
    def zero_gradients(self, rendering_params):
        self.model.zero_grad()
        
        for param in self.attacked_params:
            if param == 'textures':
                rendering_params['mesh'].textures.maps_padded().grad.zero_()
            elif param == 'mesh':
                raise NotImplementedError
            elif param in rendering_params:
                rendering_params[param].grad.zero_()
            elif param == 'shadows':
                rendering_params['shadow_image'].grad.zero_()
    
    def EOT_attack_scene(self, attacked_image, true_label=0, N_transforms=10):
        """
        This function implements Expectation Over Transformation as described in "Synthesizing Robust Adversarial Examples".
        Currently it is suited to optimizing with respect to background images and textures.
        If one needs to implement other parameter attacks, this function needs to be modified.
        """
        # Set the model to eval mode following the PyTorch tutorial for adversarial attacks
        self.model.eval()
        self.model.zero_grad()
        # pdb.set_trace()
        # Sample transformations
        rendering_params_list = [
            attacked_image.generate_rendering_params(attacked_image.get_background_image(), attacked_image.get_mesh()) for _ in range(N_transforms)
        ]
        # Make attacked parameters trainable
        rendering_params_list = [
            self.select_attacked_params(rendering_params, self.attacked_params) for rendering_params in rendering_params_list
        ]
        
        # Attack the scene
        from matplotlib import pyplot as plt
        k = 0
        while True:
            loss = torch.tensor(0, device=self.device)
            preds = []
            for rendering_params in rendering_params_list:
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
                image = torch.clip(image, min=0.0, max=1.0)
                
                # The if-statement below saves the last randomly sampled image
                if k == 0:
                    attacked_image.set_original_image(image[0])
                    attacked_image.set_rendering_params(rendering_params)
                # if k % 100 == 0:
                #     # plt.imshow(rendering_params['mesh'].textures._maps_padded[0].clone().detach().cpu().numpy())
                #     plt.imshow(image[0].permute(1, 2, 0).clone().detach().cpu().numpy())
                #     plt.savefig(f"results/tm_{k}.jpg")
                #     plt.close()
                # pdb.set_trace()
                # Run prediction on the image
                pred = self.activation(self.model(image))
                preds.append(pred.item())
                # print(f"Mean pred: {sum(preds) / len(preds)}. Min pred: {min(preds)}")
                
                # Compute the unit loss
                label_batched = torch.tensor([true_label], device=self.device, dtype=torch.float).unsqueeze(0)
                loss = loss + self.loss_fn(pred, label_batched)
                
            # Attack stops if the minimum prediction confidence goes beyond the threshold
            # The if-statement below saves the last randomly sampled image with attacked parameters
            print(min(preds))
            if min(preds) > 0.5:
                attacked_image.set_adversarial_image(image[0])
                attacked_image.set_adversarial_rendering_params(rendering_params)
                self.adversarial_examples_list.append(attacked_image)
                break

            # Propagate the gradients
            loss.backward()

            # Call FGSM attack
            rendering_params = self.fgsm_attack(rendering_params)

            # Zero all gradients
            self.zero_gradients(rendering_params)
            
            k += 1
    
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
            texture_difference_image = attacked_image.get_texture_difference_image()
            
            # Save each image to the relevant folder
            original_save_path = os.path.join(self.save_dir, "original", f"image_{idx}.jpg")
            adversarial_save_path = os.path.join(self.save_dir, "positive", f"image_{idx}.jpg")
            texture_difference_path = os.path.join(self.save_dir, "difference", f"image_{idx}.jpg")
            
            # Convert the images to PIL
            original_image = self.converter(original_image)
            adversarial_image = self.converter(adversarial_image)
            texture_difference_image = self.converter(texture_difference_image)
            
            # Save the images with high quality
            original_image.save(original_save_path, quality=95)
            adversarial_image.save(adversarial_save_path, quality=95)
            texture_difference_image.save(texture_difference_path, quality=95)
            
            idx += 1
            
    def get_num_pairs(self):
        return len(self.adversarial_examples_list)