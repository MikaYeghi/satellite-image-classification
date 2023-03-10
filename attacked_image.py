import torch
import copy
from random import uniform

from utils import sample_random_elev_azimuth, get_lightdir_from_elaz

import pdb

class AttackedImage:
    def __init__(self, background_image, mesh, device='cuda'):
        self.device = device
        self.rendering_params = self.generate_rendering_params(background_image, mesh)
        self.original_image = None
        self.adversarial_image = None
        self.adversarial_rendering_params = None
        
    def generate_rendering_params(self, background_image, mesh):
        # Camera pose
        distance = torch.tensor(5.0, device=self.device)
        elevation, azimuth = sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0)
        scaling_factor = torch.tensor(uniform(0.70, 0.80), device=self.device)
        elevation = torch.tensor(elevation, device=self.device)
        azimuth = torch.tensor(azimuth, device=self.device)

        # Lights direction and intensity
        lights_direction = get_lightdir_from_elaz(elev=uniform(0, 90), azim=uniform(-180, 180), device=self.device).clone()
        intensity = torch.tensor(uniform(0.1, 2.0), device=self.device)
        ambient_color = torch.tensor(((0.05, 0.05, 0.05),), device=self.device)

        # Collect in a dict
        rendering_params = {
            "mesh": mesh,
            "background_image": background_image,
            "distance": distance,
            "elevation": elevation,
            "azimuth": azimuth,
            "scaling_factor": scaling_factor,
            "lights_direction": lights_direction,
            "intensity": intensity,
            "ambient_color": ambient_color
        }

        return rendering_params
    
    def set_original_image(self, image):
        self.original_image = image.clone().cpu()
        
    def set_adversarial_image(self, image):
        self.adversarial_image = image.clone().cpu()
    
    def set_adversarial_rendering_params(self, rendering_params):
        self.adversarial_rendering_params = copy.deepcopy(rendering_params)
        
    def set_rendering_params(self, rendering_params):
        self.rendering_params = copy.deepcopy(rendering_params)
    
    def get_original_image(self):
        return self.original_image
    
    def get_adversarial_image(self):
        return self.adversarial_image
    
    def get_texture_difference_image(self):
        texture_difference = self.rendering_params['mesh'].textures._maps_padded - self.adversarial_rendering_params['mesh'].textures._maps_padded
        texture_difference = torch.sqrt(torch.sum(torch.square(texture_difference), dim=3))[0]
        return texture_difference
    
    def get_rendering_params(self):
        return self.rendering_params
    
    def get_background_image(self):
        return self.rendering_params['background_image']
    
    def get_mesh(self):
        return self.rendering_params['mesh']