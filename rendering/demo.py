import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

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

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

import pdb
k = 0

def savefig(img):
    global k
    img = img.clone().cpu().numpy()
    plt.savefig(os.path.join(plots_path, f"{k}.png"))
    plt.close('all')
    k += 1
    
# Set paths
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "meshes/car/car.obj")
plots_path = os.path.join(DATA_DIR, "plots")
background_images_path = os.path.join(DATA_DIR, "background_images")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

# Load the background images
background_image = Image.open(os.path.join(background_images_path, "background.png"))
background_image = pil_to_tensor(background_image)
background_image = background_image.permute(1, 2, 0)[..., :3]
background_image = background_image.clone().float().to(device) / 255

# Rotate the object by increasing the elevation and azimuth angles
sf = 0.4
R, T = look_at_view_transform(dist=5.0, elev=70, azim=-150)
cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=((sf, sf, sf),))

raster_settings = RasterizationSettings(
    image_size=50, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Move the light location so the light is shining on the cow's face.
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
lights = DirectionalLights(device=device, direction=((0,-1,0),))

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings,
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights,
        blend_params=BlendParams(background_color=background_image)
    )
)

# Re render the mesh, passing in keyword arguments for the modified components.
images = renderer(mesh, lights=lights, cameras=cameras)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");
savefig(images)
