from typing import Tuple
from collections import namedtuple
import os
import numpy as np
from omegaconf import DictConfig
import matplotlib.pyplot as plt   
import numpy as np
import random  

import torch
from torch import Tensor
import torch.nn as nn
import lightning as L

from models import ProteinDiffusion


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_iterations, lr_decay_iters=None, min_lr=1e-5, decay_lr=True):
        self.warmup = warmup_steps
        self.max_num_iters = max_iterations
        self.decay_lr = decay_lr
        self.min_lr_coeff = 0.1  
        # use lr_decay_iters< max_iterations to start constant lr phase before max_iterations
        self.lr_decay_iters = lr_decay_iters if lr_decay_iters is not None else max_iterations
        self.min_lr = min_lr  # if None uses  min_lr_coeff of peak lr

        super().__init__(optimizer)

    def get_lr(self):

        def get_min_lr(base_lr):
            return self.min_lr if self.min_lr is not None else base_lr * self.min_lr_coeff

        if not self.decay_lr:
            return list(self.base_lrs)

        if self.last_epoch < self.warmup:
            # Linear warmup phase
            return [base_lr * self.last_epoch / self.warmup for base_lr in self.base_lrs]
        elif self.last_epoch > self.lr_decay_iters:
            # Constant learning rate phase
            return [get_min_lr(base_lr) for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            decay_ratio = (self.last_epoch - self.warmup) / (self.lr_decay_iters - self.warmup)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1 + np.cos(np.pi * decay_ratio))
            return [(get_min_lr(base_lr)) + coeff * (base_lr - get_min_lr(base_lr)) for base_lr in self.base_lrs]


class ProteinModel(L.LightningModule):
    """
    Lightning wrapper class for the model, optimizer, and scheduler.
    """
    def __init__(
        self,
        backbone: ProteinDiffusion,
        weight_decay: float = 0.01,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 247,
        lr_decay_iters: int = 1000,
        max_iterations: int = 1000,
        save_dir: str = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.lr_decay_iters = lr_decay_iters
        self.weight_decay = weight_decay
        self.max_iterations = max_iterations
        self.save_dir = save_dir

    @classmethod
    def from_backbone_and_config(
        cls,
        backbone: ProteinDiffusion,
        optimizer_config: DictConfig = None,
        scheduler_config: DictConfig = None,
        save_dir: str = None,
    ):
        """
        Create a LanguageModel instance from a backbone model and configuration.
        """
        instance = cls(
            backbone=backbone,
            weight_decay=optimizer_config["weight_decay"],
            lr=optimizer_config["lr"],
            betas=optimizer_config["betas"],
            warmup_steps=scheduler_config["warmup_steps"],
            lr_decay_iters=scheduler_config["lr_decay_iters"],
            save_dir=save_dir,
        )
        # instance.apply(instance._init_weights)
        return instance


    def build_optimizer(self):  
        """  
        Create the optimizer using provided config values.  
        """  
        decay_params = []  
        no_decay_params = []  

        for name, param in self.named_parameters():  
            # Apply weight decay to Linear and Conv1d layers (excluding biases)  
            if isinstance(self.get_layer(name), (torch.nn.Linear)) and "bias" not in name:  
                decay_params.append(param)  
            else:  
                no_decay_params.append(param)  

        optimizer = torch.optim.Adam(  
            [  
                {"params": decay_params, "weight_decay": self.weight_decay},  
                {"params": no_decay_params, "weight_decay": 0.0},  
            ],  
            lr=self.lr,         # from external config  
            betas=self.betas,   # from external config  
        )  
        return optimizer  

    def build_scheduler(self, optimizer):  
        """  
        Create the learning rate scheduler with the given config.  
        """  
        scheduler = CosineWarmupScheduler(  
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            max_iterations=self.max_iterations,
            lr_decay_iters=self.lr_decay_iters, 
        )  
        return scheduler  

    def configure_optimizers(self):  
        """  
        Called by Lightning after the model is instantiated.  
        Lightning calls this method after __init__ before starting training.  
        """  
        optimizer = self.build_optimizer()  
        scheduler = self.build_scheduler(optimizer)  

        return {  
            "optimizer": optimizer,  
            "lr_scheduler": {  
                "scheduler": scheduler,  
                "interval": "step",  
            },  
        }  
    # Helper method to retrieve layer from parameter name
    def get_layer(self, param_name):
        module_names = param_name.split(".")[:-1]  # Extract module name from parameter name
        module = self
        for name in module_names:
            module = getattr(module, name)
        return module

    def forward(
        self, batch_input
    ):
        out = self.backbone(**batch_input)
        loss = out.loss
        distance_map_loss = out.distance_map_loss
        denoised_atom_pos = out.denoised_atom_pos
        diffusion_loss = out.diffusion_loss
        ModelOutput = namedtuple("ModelOutput", ["loss", "distance_map_loss", "denoised_atom_pos", "diffusion_loss"])
        return ModelOutput(loss, distance_map_loss, diffusion_loss, denoised_atom_pos)

    def training_step(self, batch_input, batch_idx) -> Tensor:

        loss, distance_map, diffusion_loss, _ = self(batch_input)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])
        self.log("train/distance_map_loss", distance_map, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])
        self.log("train/defusion_loss", diffusion_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])
        if batch_idx == 0:
            self.sample_and_visualize(batch_input, batch_idx, "train", self.save_dir)
        return loss

    def validation_step(self, batch_input, batch_idx):

        loss, distance_map_loss, diffusion_loss, _  = self(batch_input)
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])
        self.log("val/distance_map_loss", distance_map_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])
        self.log("val/defusion_loss", diffusion_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])

        if batch_idx == 0:
            self.sample_and_visualize(batch_input, batch_idx, "val", self.save_dir)


    def test_step(self, batch_input, batch_idx):
        loss, distance_map_loss, diffusion_loss, _  = self(batch_input)
        self.log("test/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])
        self.log("test/distance_map_loss", distance_map_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])
        self.log("test/defusion_loss", diffusion_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_input["atom_inputs"].shape[0])

        if batch_idx == 0:
            self.sample_and_visualize(batch_input, batch_idx, "test", self.save_dir, num_samples=10)

    def _init_weights(self, m: nn.Module, initializer_range=0.02):
        classname = m.__class__.__name__
        if classname.find("AdaptiveEmbedding") != -1:
            if hasattr(m, "emb_projs"):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i].weight, 0.0, self.emb_init_std)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, std=initializer_range)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                nn.init.xavier_normal_(m.cluster_weight)
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                nn.init.constant_(m.cluster_bias, 0.0)
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i].weight, 0.0, self.emb_init_std)
    @staticmethod
    def center_positions(positions, mask):  
        """  
        Centers the 3D positions of atoms based on the atom mask.  
        
        :param positions: numpy array of shape (m, 3), where `m` is the number of atoms.  
                        This contains the 3D positions of the atoms.  
        :param mask: numpy array of shape (m,), a boolean mask indicating valid atoms.  
        :return: Centered 3D positions of the atoms as a numpy array of shape (m, 3).  
        """  
        # Apply the mask to get only valid positions  
        valid_positions = positions[mask]  
        
        # Compute the mean position of valid atoms along each axis (x, y, z)  
        mean_position = valid_positions.mean(axis=0)  # Shape: (3,)  
        
        # Subtract the mean position from all positions to center them  
        centered_positions = positions - mean_position  
        
        return centered_positions

    @torch.inference_mode()  
    def sample_and_visualize(self, batch_input, batch_idx, mode, file_path_dir=None, num_samples=3):


        batch_sampled_atom_pos = self.backbone(**batch_input, return_loss=False).denoised_atom_pos
        # create output directory based on the mode
        if file_path_dir is None:
            file_path_dir = os.getcwd()
    
        samples_output_dir = os.path.join(file_path_dir, f"{mode}_samples")  
        os.makedirs(samples_output_dir, exist_ok=True)  
        
        # Randomly select three indices from the batch (assuming the batch size is at least 3)  
        selected_indices = random.sample(range(len(batch_sampled_atom_pos)), k=min(num_samples, len(batch_sampled_atom_pos)))  
        # selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        selected_indices = [0, 1, 2, 3]
        plt.rcParams.update({'font.size': 14})
        for example_idx in selected_indices: 
            atom_mask = batch_input["atom_mask"][example_idx].cpu().numpy() 
            sampled_atom_positions = batch_sampled_atom_pos[example_idx].cpu().numpy()
            sampled_atom_positions = self.center_positions(sampled_atom_positions, atom_mask)

            sampled_atom_positions = sampled_atom_positions[atom_mask]  
            ground_truth_atom_positions = batch_input["atom_pos"][example_idx].cpu().numpy() 
            ground_truth_atom_positions = ground_truth_atom_positions[atom_mask] 
            exm_name = f"sampled-epoch-{self.current_epoch}-batch-{batch_idx}-example-{example_idx}"  
    
            # Create a 3D plot
              
            fig = plt.figure(figsize=(10, 8))  
            ax = fig.add_subplot(111, projection='3d')  
    
            # Plot ground truth atom positions  
            ax.scatter(ground_truth_atom_positions[:, 0], ground_truth_atom_positions[:, 1],  
                    ground_truth_atom_positions[:, 2], color='blue', label='Ground Truth')  
    
            # Plot sampled atom positions  
            ax.scatter(sampled_atom_positions[:, 0], sampled_atom_positions[:, 1],  
                    sampled_atom_positions[:, 2], color='red', alpha=0.6, label='Sampled')  
    
            # Label the axes  
            ax.set_xlabel('X Coordinate')  
            ax.set_ylabel('Y Coordinate')  
            ax.set_zlabel('Z Coordinate')  
            ax.legend()  
    
            # Save the plot to a file  
            plot_filename = os.path.join(samples_output_dir, f"{exm_name}.png")  
            plt.savefig(plot_filename)  
            plt.close(fig)  # Close the figure to free memory   
