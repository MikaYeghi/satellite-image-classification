import torch
from torch import nn
import torchvision.models as models
import glob
from pytorch3d.io import load_objs_as_meshes

from renderer import Renderer

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
        self.lights_direction = torch.nn.Parameter(torch.tensor([0.0,-1.0,0.0], device=device, requires_grad=True).unsqueeze(0))
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