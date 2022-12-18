import torch
from torch import nn
from torch.nn import BCELoss
from torchvision.utils import save_image

from renderer import Renderer

import pdb

class FGSMAttacker:
    def __init__(self, model, attacked_params, epsilon=1e-3, device='cuda'):
        self.model = model
        self.attacked_params = attacked_params
        self.epsilon = epsilon
        self.device = device
        self.renderer = Renderer(device=device)
        
        # TO DO: NEED A LIST OF ORIGINAL AND ADVERSARIAL EXAMPLES
        
        # Set up the model
        self.activation = nn.Sigmoid() # Need this to apply to the model output
        self.loss_fn = BCELoss()
    
    def attack_single_image(self, rendering_params, true_label=0):
        # Set the model to eval mode following the PyTorch tutorial for adversarial attacks
        self.model.eval()
        
        # Set up parameters which require gradients for this attack
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
                save_image(image, "results/image_initial.jpg")
            
            # Run prediction on the image
            pred = self.activation(self.model(image))
            print(f"Prediction confidence: {pred.item()}")
            
            # If the class is incorrect, then stop. Otherwise, perform a single attack step
            if (pred > 0.5).int().item() != true_label:
                save_image(image, "results/image_final.jpg")
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
        for param in rendering_params:
            if param in self.attacked_params:
                # Collect the element-wise sign of the data gradient
                sign_grad = rendering_params[param].grad.data.sign()
                
                # Perturb the data
                rendering_params[param].data = rendering_params[param].data + self.epsilon * sign_grad
        
        return rendering_params
    
    def select_attacked_params(self, rendering_params, attacked_params):
        for param in rendering_params:
            if param in attacked_params:
                # NEED AN ANTIDURAK FOR MESH
                rendering_params[param].requires_grad = True
                rendering_params[param].retain_grad()

        return rendering_params
    
    def zero_gradients(self, rendering_params):
        self.model.zero_grad()
        
        for param in rendering_params:
            if param in self.attacked_params:
                rendering_params[param].grad.zero_()
    
    def __str__(self):
        text = f"Epsilon: {self.epsilon}.\nDevice: {self.device}."
        return text