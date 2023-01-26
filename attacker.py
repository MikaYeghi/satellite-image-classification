import os
import torch
import random
import torchvision
from torch import nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import join_meshes_as_batch

from renderer import Renderer
from losses import BCELoss, ColorForce, BCEColor, ClassificationScore, TVCalculator
from utils import blend_images, load_descriptive_colors, load_meshes, sample_random_elev_azimuth

import pdb

class BaseAttacker:
    def __init__(self, model, cfg, device='cuda'):
        self.model = model
        self.cfg = cfg
        self.attacked_params = cfg.ATTACKED_PARAMS
        self.device = device
        self.renderer = Renderer(device=device)
        
        # Create the save directories
        self.save_dir = cfg.ADVERSARIAL_SAVE_DIR
        Path(os.path.join(self.save_dir, "original")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_dir, "positive")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_dir, "negative")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_dir, "difference")).mkdir(parents=True, exist_ok=True)
        
        # Set up the model
        self.activation = nn.Sigmoid() # Need this to apply to the model output

class FGSMAttacker(BaseAttacker):
    def __init__(self, model, cfg, device='cuda'):
        super().__init__(model, cfg, device)
        
        # Set up attack parameters
        self.epsilon = cfg.ATTACK_LR
        
        # Load dataset descriptive colors
        self.descriptive_colors = load_descriptive_colors(cfg.DESCRIPTIVE_COLORS_PATH)
        
        # torch.Tensor to PIL.Image converter
        self.converter = torchvision.transforms.ToPILImage()
        
        # List which stores AttackedImage variables
        self.adversarial_examples_list = []
        
        # Set up the attack loss function
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
            elif param == 'pixelated-textures':
                grad_tensor = rendering_params['mesh'].textures.maps_padded().grad.data.clone()
                noise = torch.randn(size=grad_tensor.shape, device=self.device)
                noise /= torch.norm(noise, p=2)
                grad_tensor += 0.1 * noise
                colors_tensor = rendering_params['mesh'].textures.maps_padded().data.clone()
                updated_colors_tensor = torch.empty(size=colors_tensor.shape[1:], device=self.device)
                downsampled_size = 512 // self.cfg.ATTACKED_PIXELATED_TEXTURE_BLOCK_SIZE
                errors_tensors = torch.empty(size=(0, downsampled_size, downsampled_size), device=self.device)
                for descriptive_color in self.descriptive_colors:
                    # Colors are in a 3D space. Find the descriptive color 
                    # which is closest to the gradient direction. This is done using simple vector
                    # manipulations.
                    lambda_color = torch.sum((descriptive_color - colors_tensor) * grad_tensor, dim=-1)
                    lambda_color = lambda_color / (torch.sum(grad_tensor * grad_tensor, dim=-1) + 1e-6)
                    errors_color = descriptive_color - colors_tensor - torch.mul(lambda_color.unsqueeze(-1), grad_tensor)
                    errors_color = torch.norm(errors_color, p=2, dim=-1)
                    errors_tensors = torch.cat((errors_tensors, errors_color), dim=0)
                errors_tensors += 1e6 * (errors_tensors == 0).float() # Current color point has error 0, causing no change
                color_indices_tensor = torch.argmin(errors_tensors, dim=0)
                for i in range(color_indices_tensor.shape[0]):
                    for j in range(color_indices_tensor.shape[1]):
                        updated_colors_tensor[i][j] = self.descriptive_colors[color_indices_tensor[i][j]]
                rendering_params['mesh'].textures.maps_padded().data = updated_colors_tensor.unsqueeze(0)
                # print(f"Average change: {torch.norm(updated_colors_tensor - colors_tensor, p=2).item()}")
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
            elif attacked_param == 'pixelated-textures':
                for i in range(len(rendering_params['mesh'].textures.maps_list())):
                    downsampled_size = 512 // self.cfg.ATTACKED_PIXELATED_TEXTURE_BLOCK_SIZE
                    rendering_params['mesh'].textures.maps_padded().data = torch.rand(size=(1, downsampled_size, downsampled_size, 3), device=self.device)
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
            if param == 'textures' or param == 'pixelated-textures':
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
                print(f"Mean pred: {sum(preds) / len(preds)}. Min pred: {min(preds)}")
                
                # Compute the unit loss
                label_batched = torch.tensor([true_label], device=self.device, dtype=torch.float).unsqueeze(0)
                loss = loss + self.loss_fn(pred, label_batched)
                
            # Attack stops if the minimum prediction confidence goes beyond the threshold
            # The if-statement below saves the last randomly sampled image with attacked parameters
            # print(min(preds))
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
    
class UnifiedTexturesAttacker(BaseAttacker):
    def __init__(self, model, attack_set, cfg, device='cuda'):
        super().__init__(model, cfg, device)
        
        # Load the meshes
        self.cfg = cfg
        self.attack_set = self.prepare_attack_set(attack_set)
        self.meshes = load_meshes(cfg, device='cpu')
        self.device = device
        
        # Set up the attack loss function
        self.loss_fns = self.get_loss_fns(cfg)
    
    def __str__(self):
        text = "-" * 80 + "\n"
        text += "UnifiedTexturesAttacker.\n"
        text += f"Model: {self.model.__class__.__name__}.\n"
        text += f"Number of meshes: {len(self.meshes)} vehicles.\n"
        text += f"Dataset size: {len(self.attack_set)} empty images.\n"
        text += "-" * 80
        return text
    
    def prepare_attack_set(self, attack_set):
        """
        Preprocess the dataset which is used for attacking the model.
        Need to remove positive samples, since images are generated on the fly during the attack.
        
        inputs:
            - attack_set: an object of class SatelliteDataset
        outputs:
            - attack_set: an object of class SatelliteDataset without positive samples
        """
        attack_set.remove_positives()
        return attack_set
    
    def get_loss_fns(self, cfg):
        """
        Extract loss function terms from the loss function keyword.
        
        inputs:
            - cfg: configurations loaded from the config file
        outputs:
            - loss_fns_list: list of loss function terms
        """
        loss_fn_keyword = cfg.ATTACK_LOSS_FUNCTION
        loss_fn_parameters = cfg.ATTACK_LOSS_FUNCTION_PARAMETERS
        loss_fns_dict = {}
        if "classcore" in loss_fn_keyword:
            loss_fn = ClassificationScore(class_id=loss_fn_parameters['classcore'], coefficient=loss_fn_parameters['classcore-coefficient'])
            loss_fns_dict['classcore'] = loss_fn
        if "TV" in loss_fn_keyword:
            loss_fn = TVCalculator(coefficient=loss_fn_parameters['TV-coefficient'])
            loss_fns_dict['TV'] = loss_fn
        return loss_fns_dict
    
    def attack(self):
        """
        Perform adversarial attack to obtain a unified adversarial texture map.
        """
        # Initialize the attacked texture map as a random map
        textures_activation = torch.nn.Sigmoid()
        adv_textures = torch.randn(size=self.meshes[0].textures.maps_padded().shape, device=self.device)
        adv_textures = textures_activation(adv_textures)
        adv_textures.requires_grad_(True)
        
        # Dataloader
        attack_set_loader = DataLoader(self.attack_set, batch_size=self.cfg.ATTACK_BATCH_SIZE, shuffle=True)
        
        # Set up the attack optimization
        optimizer = torch.optim.Adam([adv_textures], lr=self.cfg.ATTACK_BASE_LR, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.ATTACK_LR_GAMMA)
        
        # Logging
        writer = SummaryWriter(log_dir=self.cfg.LOG_DIR)
        iter_counter = 0
        
        # Perform the attack
        for epoch in range(self.cfg.ATTACK_N_EPOCHS):
            progress_bar = tqdm(attack_set_loader, desc=f"Epoch #{epoch + 1}")
            for empty_images_batch, _ in progress_bar:
                # Generate an image with a vehicle in it
                images_batch = self.render_images_batch(empty_images_batch, adv_textures, centered=self.cfg.CENTERED_IMAGES_ATTACK)
                
                # Generate predictions
                self.model.train()
                preds = self.model(images_batch)
                preds = self.activation(preds)
                loss_dict = self.loss_dict_forward(preds, adv_textures)
                total_loss = sum(loss_dict.values())
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Clamp to the image range
                adv_textures.data.clamp_(0, 1)
                
                # Log
                for loss_fn in loss_dict:
                    writer.add_scalar(loss_fn, loss_dict[loss_fn].item(), iter_counter)
                writer.add_scalar("Total loss", total_loss.item(), iter_counter)
                
                # Update the progress bar
                progress_bar.set_description(f"Epoch: #{epoch + 1}. Loss: {total_loss.item()}")
                
                # Update the iterations counter
                iter_counter += 1
            
            # Update learning rate
            scheduler.step()
            
            # Save the texture map
            save_image(adv_textures[0].permute(2, 0, 1), f"results/adv_textures_{epoch}.png")
        
        # Close the tensorboard logger
        writer.flush()
        writer.close()
        
        # Save the unified texture map [TO DO: CHANGE THE SAVE DIR TO ADVERSARIAL_ATTACKS_SAVE_DIR]
        transform_tensor_to_pil = torchvision.transforms.ToPILImage()
        adv_textures_PIL = transform_tensor_to_pil(adv_textures[0].permute(2, 0, 1))
        adv_textures_PIL.save(os.path.join(self.cfg.OUTPUT_DIR, "unified_adversarial_textures.png"))
        
    def render_image(self, empty_image, adv_textures, centered=True):
        """
        Render an image, where the vehicle has the unified texture map (adv_textures).
        
        inputs:
            - empty_image: a background image
            - adv_textures: the unified texture map which is placed on every mesh
        outputs:
            - image: an image with vehicles
        """
        # Randomly select a mesh from the list of meshes
        mesh = random.choice(self.meshes).to(self.device)
        
        # Replace the texture map
        mesh.textures._maps_padded = adv_textures
        
        # Offset the vehicle if non-centered
        if centered:
            pass
        else:
            raise NotImplementedError
        
        # Sample rendering parameters
        distance = 5.0
        elevation, azimuth = sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0) 
        lights_direction = torch.tensor([random.uniform(-1, 1), -1.0 ,random.uniform(-1, 1)], device=self.device, requires_grad=True).unsqueeze(0)
        scaling_factor = random.uniform(0.70, 0.80)
        intensity = random.uniform(0.2, 2.0)
        ambient_color = ((0.05, 0.05, 0.05),)
        
        # Render the image
        image = self.renderer.render(
                mesh=mesh,
                background_image=empty_image,
                elevation=elevation,
                azimuth=azimuth,
                lights_direction=lights_direction,
                distance=distance,
                scaling_factor=scaling_factor,
                intensity=intensity,
                ambient_color=ambient_color
        ).permute(2, 0, 1).unsqueeze(0)
        
        return image
    
    def render_images_batch(self, empty_images_batch, adv_textures, centered=True):
        batch_size = len(empty_images_batch)
        
        # Randomly select a mesh from the list of meshes
        meshes = random.sample(self.meshes, len(empty_images_batch))
        meshes = join_meshes_as_batch(meshes).to(self.device)
        
        # Replace the texture maps
        meshes.textures._maps_padded = adv_textures.repeat(batch_size, 1, 1, 1)
        
        # Offset the vehicle if non-centered
        if centered:
            pass
        else:
            raise NotImplementedError
        
        # Sample rendering parameters
        distances = [5.0] * batch_size
        els_azs = [sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0) for _ in range(batch_size)]
        elevations = [els_azs_[0] for els_azs_ in els_azs]
        azimuths = [els_azs_[1] for els_azs_ in els_azs]
        light_directions = torch.empty(size=(0, 3), device=self.device)
        for _ in range(batch_size):
            light_direction = torch.tensor([random.uniform(-1, 1), -1.0 ,random.uniform(-1, 1)], device=self.device).unsqueeze(0)
            light_directions = torch.cat((light_directions, light_direction))
        scaling_factors = torch.ones(size=(batch_size, 3), device=self.device)
        for i in range(batch_size):
            scaling_factors[i] *= random.uniform(0.70, 0.80)
        intensities = torch.ones(size=(batch_size, 3), device=self.device)
        for i in range(batch_size):
            intensities[i] *= random.uniform(0.70, 0.80)
        ambient_color = ((0.05, 0.05, 0.05),)
        
        # Render the image
        images = self.renderer.render_batch(
                meshes=meshes,
                background_images=empty_images_batch,
                elevations=elevations,
                azimuths=azimuths,
                light_directions=light_directions,
                distances=distances,
                scaling_factors=scaling_factors,
                intensities=intensities,
                ambient_color=ambient_color
        )
        
        return images
    
    def loss_dict_forward(self, predictions, adversarial_textures):
        loss_dict = {}
        for loss_fn in self.loss_fns:
            loss_dict[loss_fn] = self.loss_fns[loss_fn](predictions, adversarial_textures)
        return loss_dict