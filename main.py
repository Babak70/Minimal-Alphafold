import os
from datetime import datetime  
import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from data_utils import ProteinDataModule
from models import ProteinDiffusion
from PLmodel import ProteinModel

# add dataclass
from dataclasses import dataclass, asdict
from utils import load_model_from_checkpoint

# tot train size is 18000 examples
# with 64 batch size, each epoch is 282 steps

TOTAL_TRAIN_SIZE = 18000
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_STEPS_PER_EPOCH = TOTAL_TRAIN_SIZE // BATCH_SIZE
MAX_STEPS = NUM_EPOCHS * NUM_STEPS_PER_EPOCH

print(f"total train size: {TOTAL_TRAIN_SIZE}")
print(f"batch size: {BATCH_SIZE}")
print(f"num epochs: {NUM_EPOCHS}")
print(f"num steps per epoch: {NUM_STEPS_PER_EPOCH}")
print(f"max steps: {MAX_STEPS}")


  
 
current_dir = os.getcwd()  
datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
save_dir = os.path.join(current_dir, datetime_str)  
os.makedirs(save_dir, exist_ok=True)  

@dataclass  
class TrainerConfig:  
    accelerator: str = 'gpu'  
    devices: str = 'auto'  
    precision: str = '32'  
    gradient_clip_val: float = 10.0  
    max_steps: int = MAX_STEPS  
    log_every_n_steps: int = 5  
    val_check_interval: int = 0.25
    enable_checkpointing: bool = True  
    enable_model_summary: bool = True  
    use_distributed_sampler: bool = True 


@dataclass
class OptimizerConfig:  
    lr: float = 1e-3  
    weight_decay: float = 0.00
    betas: tuple = (0.9, 0.999)

@dataclass
class SchedulerConfig:  
    warmup_steps: int = 100  
    max_iterations: int = MAX_STEPS
    lr_decay_iters: int = int(MAX_STEPS * 0.9)
  

@dataclass  
class Config:  
    max_seq_length: int = 256  
    batch_size: int = BATCH_SIZE
    num_sample_steps: int = 50
    batch_size_eval: int = 4
    dim_atom_inputs: int = 77
    dim_atom_inputs_init: int = 3  
    default_num_molecule_mods: int = 5  
    predict: bool = False 
    ckpt_path: str = None
    # ckpt_path: str = "/home/t-brahmani/babak/alphafold3-pytorch-lightning-hydra/2025-02-14_17-33-45/epoch=19-step=5460.ckpt"
    seed: int = 42  
    trainer: TrainerConfig = TrainerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint_every_n_steps: int = NUM_STEPS_PER_EPOCH//2
    save_dir: str = save_dir
    sigma_data: float = 10.95623255143087

config = Config()

def main():
    seed_everything(config.seed)

    # Fully utilize tensor cores
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
       **asdict(config.trainer),
        callbacks=[
            ModelSummary(max_depth=2),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=10,
                every_n_train_steps=config.checkpoint_every_n_steps,
                verbose=True,
                dirpath=save_dir, # Set your custom path here  
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    datamodule = ProteinDataModule(batch_size=config.batch_size, batch_size_eval=config.batch_size_eval, max_seq_length=config.max_seq_length, dim_atom_inputs_init=config.dim_atom_inputs_init, default_num_molecule_mods=config.default_num_molecule_mods)
    backbone = ProteinDiffusion(config.dim_atom_inputs, num_sample_steps=config.num_sample_steps)

    with trainer.init_module():
        model = ProteinModel.from_backbone_and_config(backbone, optimizer_config = asdict(config.optimizer), scheduler_config= asdict(config.scheduler), save_dir=config.save_dir)

    # load model from checkpoint
    ckpt_path = config.ckpt_path
    if config.predict:
        assert ckpt_path is not None, "ckpt_path must be provided for prediction"
        model = load_model_from_checkpoint(model, ckpt_path)
        model.save_dir = os.path.dirname(ckpt_path)
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path if ckpt_path else None)
        # plot_all_metrics(trainer, save_dir)
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()