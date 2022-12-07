from torch import nn

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams
)

import pdb

class Renderer(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
    
    def render(self, mesh, background_image, distance, elevation, azimuth, lights_direction, 
               scaling_factor=0.4, image_size=50, blur_radius=0.0, faces_per_pixel=1):
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        cameras = FoVOrthographicCameras(
            device=self.device, 
            R=R, 
            T=T, 
            scale_xyz=((scaling_factor, scaling_factor, scaling_factor),)
        )
        
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=blur_radius, 
            faces_per_pixel=faces_per_pixel, 
        )
        
        lights = DirectionalLights(device=self.device, direction=lights_direction)
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=background_image)
            )
        )

        images = renderer(mesh, lights=lights, cameras=cameras)
        return images[0, ..., :3]