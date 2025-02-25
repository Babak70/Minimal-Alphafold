
from typing import Optional, NamedTuple
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.nn import Module, ModuleList
from einops import rearrange, repeat
from torch import Tensor
from einops.layers.torch import Rearrange
from tqdm import tqdm
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from utils import (
    softclamp,
    log,
)


class FourierEmbedding(Module):
    """Algorithm 22."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times: Tensor, 
    ):  
        """Perform the forward pass.

        :param times: The times tensor.
        :return: The output tensor.
        """
        times = rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return torch.cos(2 * np.pi * rand_proj)
    

class Embedding(nn.Module):
    """Class that applies an extra projection to the input embeddings if the dimensions do not match"""

    def __init__(self, n_tokens, d_embed, d_model, initializer_range=0.02):  
        super().__init__()  
        self.d_embed = d_embed  
        self.d_model = d_model  
        self.n_tokens = n_tokens  
        self.initializer_range = initializer_range  

        self.emb = nn.Embedding(n_tokens, d_embed)  

        if d_model != d_embed:  
            self.projection = nn.Linear(d_embed, d_model, bias=False)  

        self.reset_parameters()  

    def reset_parameters(self):  
        # Loop over all submodules and initialize their weights with a normal distribution.  
        for m in self.modules():  
            if hasattr(m, 'weight') and m.weight is not None:  
                nn.init.normal_(m.weight, std=self.initializer_range)  
            # Optionally, initialize biases to zero if they exist.  
            if hasattr(m, 'bias') and m.bias is not None:  
                nn.init.constant_(m.bias, 0)  

    def forward(self, x):  
        x = self.emb(x)  
        if self.d_model != self.d_embed:  
            x = self.projection(x)  
        return x  

class FeatureExpander(nn.Module):
    def __init__(self, d_model, d_embed=4,  initializer_range=0.02):  
        super().__init__()
        d_embed = d_model  
        self.d_embed = d_embed  
        self.d_model = d_model   
        self.initializer_range = initializer_range  

        self.emb = nn.Linear(1, d_embed, bias=False)

        if d_model != d_embed:  
            self.projection = nn.Linear(d_embed, d_model, bias=False)  

        self.reset_parameters()  

    def reset_parameters(self):  
        # Loop over all submodules and initialize their weights with a normal distribution.  
        for m in self.modules():  
            if hasattr(m, 'weight') and m.weight is not None:  
                nn.init.normal_(m.weight, std=self.initializer_range)  
            # Optionally, initialize biases to zero if they exist.  
            if hasattr(m, 'bias') and m.bias is not None:  
                nn.init.constant_(m.bias, 0)  

    def forward(self, x):

        x = x.unsqueeze(-1) 
        x = self.emb(x)  
        if self.d_model != self.d_embed:  
            x = self.projection(x)  
        return x  


def compute_distance_map(atom_pos: Tensor, atom_mask: Tensor) -> Tensor:  
    # Calculate the pairwise squared Euclidean distances  
    atom_diff = atom_pos.unsqueeze(2) - atom_pos.unsqueeze(1)
    distance_map = torch.sqrt(torch.sum(atom_diff ** 2, dim=-1) + 1e-8)

    # Apply mask to set distances for non-existing atoms to zero  
    mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)  
    distance_map = distance_map * mask  

    return distance_map  

def distance_map_loss(pred_distance_map: Tensor, true_distance_map: Tensor, atom_mask: Tensor) -> Tensor:  
    mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)  
    loss = F.mse_loss(pred_distance_map[mask], true_distance_map[mask], reduction='mean')  
    return loss
    

class MLPs(nn.Module):  
    def __init__(self, dim: int, expansion_factor: int=4):  
        """  .  
        """  
        super().__init__()  
        dim_inner = int(dim * expansion_factor)  
        self.ff = nn.Sequential(  
            nn.Linear(dim, dim_inner * 2, bias=False),  
            nn.Linear(dim_inner, dim, bias=False),  
        )  
  
    def forward(self, x):  
        """  
        Perform the forward pass.  
  
        Parameters:  
        - x: The input tensor.  
  
        Returns:  
        - The output tensor after applying the MLP with SwiGLU activation.  
        """  

        x, gates = self.ff[0](x).chunk(2, dim=-1)  
        output = F.silu(gates) * x  
        return self.ff[1](output)  

    

class Attention(Module):  
    def __init__(  
        self,  
        dim,  
        dim_head=64,  
        heads=8,  
        dropout=0.0,  
        gate_output=True,  
        query_bias=True,  
        num_memory_kv: int = 0,  
        enable_attn_softclamp=False,  
        attn_softclamp_value=50.0,  
        init_gate_bias=-2.0,
        apply_rotary: bool = True,
        rope_theta: float = 1024.0,
        max_seq_len: int = 1024,
    ):  
        super().__init__()  
        dim_inner = dim_head * heads  
  
        self.scale = dim_head ** -0.5  
        self.attn_dropout = nn.Dropout(dropout)  
        self.enable_attn_softclamp = enable_attn_softclamp  
        self.attn_softclamp_value = attn_softclamp_value  
  
        self.to_q = nn.Linear(dim, dim_inner, bias=query_bias)  
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)  
        self.to_out = nn.Linear(dim_inner, dim, bias=False)  
        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)  
        self.merge_heads = Rearrange("b h n d -> b n (h d)")  
  
        self.memory_kv = None  
        if num_memory_kv > 0:  
            self.memory_kv = nn.Parameter(torch.zeros(2, heads, num_memory_kv, dim_head))  
            nn.init.normal_(self.memory_kv, std=0.02)  
  
        self.to_gates = None  
        if gate_output:  
            gate_linear = nn.Linear(dim, dim_inner)  
            nn.init.zeros_(gate_linear.weight)  
            nn.init.constant_(gate_linear.bias, init_gate_bias)  
            self.to_gates = gate_linear

        # ROPE
        self.apply_rotary = apply_rotary
        self.rope_theta = rope_theta
        if self.apply_rotary:
            self.rotary_emb = LlamaRotaryEmbedding(
                dim=dim_head,
                max_position_embeddings=max_seq_len,
                base=self.rope_theta,
            )
  
    def forward(  
        self,  
        seq: Tensor,
        mask: bool = None,
        context: Tensor = None,
        attn_bias: Tensor = None,
        position_ids: Optional[torch.Tensor] = None, 
    ) -> Tensor:
        q = self.to_q(seq)  
  
        context_seq = context if context is not None else seq  
        k, v = self.to_kv(context_seq).chunk(2, dim=-1)  
  
        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))
        if self.apply_rotary:
            seq_len = k.shape[2]
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=seq.device)
                position_ids = repeat(position_ids, "T -> B T", B=seq.size(0))
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids) 
  
 
        q = q * self.scale  
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) 
        if attn_bias is not None:  
            sim = sim + attn_bias  

        if self.enable_attn_softclamp:  
            sim = softclamp(sim, self.attn_softclamp_value)  

        if mask is not None:  
            mask_value = -torch.finfo(sim.dtype).max  
            sim = torch.where(mask[:, None, None, :], sim, mask_value)  
  
        attn = sim.softmax(dim=-1)  
        attn = self.attn_dropout(attn)  

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)  
        out = self.merge_heads(out)  
  
        if self.to_gates is not None:  
            gates = self.to_gates(seq).sigmoid()  
            out = out * gates  
  
        return self.to_out(out) 


class TransformerBlock(Module):

    def __init__(
        self,
        depth,
        heads,
        dim=384,
        attn_kwargs: dict = dict(),
        attn_num_memory_kv=False,
        mlp_expansion_factor=2,
    ):
        super().__init__()

        layers = ModuleList([])

        for _ in range(depth):

            attn = Attention(
                dim=dim,
                heads=heads,
                num_memory_kv=attn_num_memory_kv,
                **attn_kwargs,
            )
            mlp = MLPs(
                dim=dim,
                expansion_factor=mlp_expansion_factor,
            )
            norm_pre_attention = nn.LayerNorm(dim, elementwise_affine=True)
            norm_pre_mlp = nn.LayerNorm(dim, elementwise_affine=True)
            layers.append(
                ModuleList(
                    [norm_pre_attention, attn, norm_pre_mlp, mlp]  # type: ignore
                )
            )
        self.layers = layers

    def forward(
        self,
        atom_feats: Tensor,
        mask: bool = None,
    ):
        for pre_norm_att, attn, pre_norm_mlp, mlp in self.layers:
            atom_feats = pre_norm_att(atom_feats)
            atom_feats = (
                attn(
                    atom_feats,
                    mask=mask,
                )
                + atom_feats
            )
            atom_feats = pre_norm_mlp(atom_feats)

            atom_feats = mlp(atom_feats) + atom_feats
        return atom_feats

class InputEmbedder(Module):

    def __init__(
        self,
        dim_atom_inputs,
        dim_atompair_inputs=5,
        dim_atom=128,
        dim_atompair=16,
        atom_transformer_blocks=3,
        atom_transformer_heads=4,
        atom_transformer_kwargs: dict = dict(),
    ):
        super().__init__()

        self.to_atom_feats = nn.Linear(dim_atom_inputs, dim_atom, bias=False)

        self.to_atompair_feats = nn.Linear(dim_atompair_inputs, dim_atompair, bias=False)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_atom),
            nn.Linear(dim_atom, dim_atompair * 2, bias=False),
            nn.ReLU(),
        )

        self.atompair_feats_mlp = nn.Sequential(
            nn.Linear(dim_atompair, dim_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(dim_atompair, dim_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(dim_atompair, dim_atompair, bias=False),
        )

        self.atom_transformer = TransformerBlock(
            depth=atom_transformer_blocks,
            heads=atom_transformer_heads,
            dim=dim_atom,
            **atom_transformer_kwargs,
        )

    def forward(
        self,
        atom_inputs: Tensor,
        atompair_inputs: Tensor,
    ):
        """
        Embed the atom and atompair inputs.
        """

        atom_feats = self.to_atom_feats(atom_inputs)
        atompair_feats = self.to_atompair_feats(atompair_inputs)

        atom_feats = self.atom_transformer(
            atom_feats
        )

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        return (
            atom_feats,
            atompair_feats,
        )

class DiffusionModule(Module):
    def __init__(
        self, 
        dim_atom=128,
        dim_atompair=16,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_encoder_kwargs: dict = dict(),
        atom_decoder_kwargs: dict = dict(),
        sigma_data= 10.9,
    ):
        super().__init__()

        self.atom_pos_to_atom_feat = nn.Linear(3, dim_atom, bias=False)
        self.time_embedding = FourierEmbedding(dim_atom)

        self.atom_encoder = TransformerBlock(
            dim=dim_atom,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            **atom_encoder_kwargs,
        )

        self.atom_decoder = TransformerBlock(
            dim=dim_atom,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            **atom_decoder_kwargs,
        )


        self.atom_pair_embedding = nn.Sequential(  
            nn.Linear(dim_atompair, dim_atompair),  
            nn.ReLU(),  
            nn.Linear(dim_atompair, dim_atompair)  
        )  
        
        # Fuse the two aggregated atompair chunks (each of size dim_atom)  
        self.fuse_atompair = nn.Linear(2 * dim_atompair, dim_atom)
        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(dim_atom), nn.Linear(dim_atom, 3, bias=False)
        )

    def forward(
        self,
        noised_atom_pos: Tensor,
        atom_feats: Tensor,
        atompair_feats: Tensor,
        atom_mask: bool,
        times: Tensor,

    ):
        """Perform the forward pass.
        """
        times_emb = self.time_embedding(times)
        noised_atom_pos_feats = self.atom_pos_to_atom_feat(noised_atom_pos)
        times_emb = repeat(times_emb, "b d -> b n d", n=noised_atom_pos_feats.shape[1])
        atom_feats = noised_atom_pos_feats + atom_feats + times_emb*0.0

 
        # 1. Embed atompair_feats from 5 -> dim_atom  
        pair_emb = self.atom_pair_embedding(atompair_feats)  # (b, seq, seq, dim_atom)  
        
        # 2. Aggregate along two different dimensions:  
 
        pair_row = pair_emb.mean(dim=2)  
 
        pair_col = pair_emb.mean(dim=1) 
         
        fused_pair = self.fuse_atompair(torch.cat([pair_row, pair_col], dim=-1))  # (b, seq, dim_atom)  
        
        # Fuse the atompair conditioning with the atom_feats.  
        atom_feats = atom_feats + fused_pair + times_emb
        # atom encoder
        atom_feats_encoded = self.atom_encoder(
            atom_feats,
            mask=atom_mask,
        )
        #  atom decoder
        atom_feats = self.atom_decoder(
            atom_feats_encoded,
            mask=atom_mask,
        )

        atom_pos_update = self.atom_feat_to_atom_pos_update(atom_feats)

        return atom_pos_update


class Diffuser(Module):
    """An Diffuser module."""

    def __init__(
        self,
        net: DiffusionModule,
        num_sample_steps=50,  # number of sampling steps
        sigma_min=0.002,  # min noise level
        sigma_max=80,  # max noise level
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
        step_scale=1.5,
    ):
        super().__init__()

        self.net = net

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps
        self.step_scale = step_scale

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        # whether to use original karras formulation or not


    def c_skip(self, sigma):
        """Return the c_skip value.

        :param sigma: The sigma value.
        :return: The c_skip value.
        """
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """Return the c_out value.

        :param sigma: The sigma value.
        :return: The c_out value.
        """
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma):
        """Return the c_in value.

        :param sigma: The sigma value.
        :return: The c_in value.
        """
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        """Return the c_noise value.

        :param sigma: The sigma value.
        :return: The c_noise value.
        """
        return log(sigma) * 0.25

    # preconditioned network output

 
    def preconditioned_network_forward(
        self,
        noised_atom_pos: Tensor,
        sigma: Tensor,
        network_condition_kwargs: dict,
        clamp=False,
    ):
        """Run a network forward pass, with the preconditioned inputs.

        :param noised_atom_pos: The noised atom position tensor.
        :param sigma: The sigma value.
        :param network_condition_kwargs: The network condition keyword arguments.
        :param clamp: Whether to clamp the output.
        :return: The output tensor.
        """
        batch, dtype, device = (
            noised_atom_pos.shape[0],
            noised_atom_pos.dtype,
            noised_atom_pos.device,
        )

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, dtype=dtype, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.net(
            self.c_in(padded_sigma) * noised_atom_pos,
            times=sigma,
            **network_condition_kwargs,
        )

        out = self.c_skip(padded_sigma) * noised_atom_pos + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out


    def sample_schedule(self, num_sample_steps=None):
        """Return the schedule of sigmas for sampling. Algorithm (7) in the paper.

        :param num_sample_steps: The number of sample steps.
        :return: The schedule of sigmas for sampling.
        """
        num_sample_steps = num_sample_steps if num_sample_steps is not None else self.num_sample_steps

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=self.dtype)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (N - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.

        return sigmas * self.sigma_data

    @torch.no_grad()
    def sample(self, atom_mask: bool = None, num_sample_steps: int = None, clamp: bool=False, return_all_timesteps:bool=False, **network_condition_kwargs):  # type: ignore
        """Sample clean atom positions.

        :param atom_mask: The atom mask tensor.
        :param num_sample_steps: The number of sample steps.
        :param clamp: Whether to clamp the output.
        :param network_condition_kwargs: The network condition keyword arguments.
        :param use_tqdm_pbar: Whether to use tqdm progress bar.
        :param tqdm_pbar_title: The tqdm progress bar title.
        :param return_all_timesteps: Whether to return all timesteps.
        :return: The clean atom positions.
        """
        dtype = self.dtype

        step_scale, num_sample_steps = self.step_scale, num_sample_steps if num_sample_steps is not None else self.num_sample_steps

        shape = (*atom_mask.shape, 3)

        network_condition_kwargs.update(atom_mask=atom_mask)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # atom position is noise at the beginning

        init_sigma = sigmas[0]

        atom_pos = init_sigma * torch.randn(shape, dtype=dtype, device=self.device)

        # gradually denoise

        all_atom_pos = [atom_pos]

        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas, desc="sampling"
        ):
            sigma, sigma_next, gamma = tuple(t.item() for t in (sigma, sigma_next, gamma))


            eps = self.S_noise * torch.randn(
                shape, dtype=dtype, device=self.device
            )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            atom_pos_hat = atom_pos + sqrt(sigma_hat**2 - sigma**2) * eps

            model_output = self.preconditioned_network_forward(
                atom_pos_hat,
                sigma_hat,
                clamp=clamp,
                network_condition_kwargs=network_condition_kwargs,
            )
            denoised_over_sigma = (atom_pos_hat - model_output) / sigma_hat

            atom_pos_next = (
                atom_pos_hat + (sigma_next - sigma_hat) * denoised_over_sigma * step_scale
            )

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    atom_pos_next,
                    sigma_next,
                    clamp=clamp,
                    network_condition_kwargs=network_condition_kwargs,
                )
                denoised_prime_over_sigma = (atom_pos_next - model_output_next) / sigma_next
                atom_pos_next = (
                    atom_pos_hat
                    + 0.5
                    * (sigma_next - sigma_hat)
                    * (denoised_over_sigma + denoised_prime_over_sigma)
                    * step_scale
                )

            atom_pos = atom_pos_next

            all_atom_pos.append(atom_pos)

        if return_all_timesteps:
            atom_pos = torch.stack(all_atom_pos)

        if clamp:
            atom_pos = atom_pos.clamp(-1.0, 1.0)

        return atom_pos

    # training

    @property
    def device(self):
        """Return the device of the module.

        :return: The device of the module.
        """
        return next(self.net.parameters()).device

    @property
    def dtype(self):
        """Return the dtype of the module.

        :return: The dtype of the module.
        """
        return next(self.net.parameters()).dtype

    def karras_loss_weight(self, sigma):
        """Return the loss weight for training.

        :param sigma: The sigma value.
        :return: The loss weight for training.
        """
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def loss_weight(self, sigma):
        """Return the loss weight for training. For some reason, in paper they add instead of
        multiply as in original paper.

        :param sigma: The sigma value.
        :return: The loss weight for training.
        """
        return (sigma**2 + self.sigma_data**2) * (sigma + self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        """Sample Gaussian-distributed noise.

        :param batch_size: The batch size.
        :return: Sampled Gaussian noise.
        """
        return (
            self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)
        ).exp() * self.sigma_data

    def forward(
        self,
        atom_pos_ground_truth: Tensor,
        atom_mask: bool,
        atom_feats: Tensor,
        atompair_feats: Tensor,
    ):


        dtype = atom_pos_ground_truth.dtype
        batch_size = atom_pos_ground_truth.shape[0]

        sigmas = self.noise_distribution(batch_size).type(dtype)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        noise = torch.randn_like(atom_pos_ground_truth)

        noised_atom_pos = (
            atom_pos_ground_truth + padded_sigmas * noise
        )

        denoised_atom_pos = self.preconditioned_network_forward(
            noised_atom_pos,
            sigmas,
            network_condition_kwargs=dict(
                atom_feats=atom_feats,
                atom_mask=atom_mask,
                atompair_feats=atompair_feats,
            ),
        )

        losses = (
            F.mse_loss(
                denoised_atom_pos,
                atom_pos_ground_truth,
                reduction="none",
            )
            / 3.0
        )

        loss_weights = self.karras_loss_weight(padded_sigmas)
        losses = losses * loss_weights
        mse_loss = losses[atom_mask].mean()


        return mse_loss, denoised_atom_pos, sigmas

class ProteinDiffusionOutput(NamedTuple):
    loss: Optional[float] = None
    denoised_atom_pos: Optional[Tensor] = None
    sigmas: Optional[Tensor] = None
    distance_map_loss: Optional[Tensor] = None
    diffusion_loss: Optional[Tensor] = None

class ProteinDiffusion(Module):
    def __init__(
        self,
        dim_atom_inputs,
        dim_atom=128,
        emb_dim=77,
        n_tokens=22,
        dim_atompair_inputs=5,
        dim_atompair=16,
        sigma_data=10.9,
        num_sample_steps=None, 
        input_embedder_kwargs: dict = dict(
            atom_transformer_blocks=3,
            atom_transformer_heads=4,
            atom_transformer_kwargs=dict(),
        ),
        diffusion_module_kwargs: dict = dict(
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            atom_decoder_depth=3,
            atom_decoder_heads=4,
        ),
        diffuser_kwargs: dict = dict(
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            P_mean=-1.2,
            P_std=1.2,
            S_churn=80,
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
        ),
        distance_loss_weight: float = 0.01,
    ):
        super().__init__()

        self.num_sample_steps = num_sample_steps
        # initial linear layer from amino-acid to word embedding
        # self.atom_inputs_to_atom_feats = nn.Linear(3, dim_atom_inputs, bias=False)
        self.atom_inputs_to_atom_feats = FeatureExpander(dim_atom_inputs)
        # self.atom_inputs_to_atom_feats = Embedding(n_tokens, emb_dim, dim_atom_inputs)
        self.distance_loss_weight = distance_loss_weight

        # input feature embedder
        self.input_embedder = InputEmbedder(
            dim_atom_inputs=dim_atom_inputs,
            dim_atompair_inputs=dim_atompair_inputs,
            dim_atom=dim_atom,
            dim_atompair=dim_atompair,
            **input_embedder_kwargs,
        )

        # Diffusion block denoiser + conditioner
        self.diffusion_module = DiffusionModule(
            sigma_data=sigma_data,
            dim_atom=dim_atom,
            dim_atompair=dim_atompair,
            **diffusion_module_kwargs,
        )
        # main diffusion module
        self.diffuser = Diffuser(
            self.diffusion_module,
            sigma_data=sigma_data,
            **diffuser_kwargs,
        )
    def compute_loss_with_distance_map(  
        self,  
        denoised_atom_pos: Tensor,  
        atom_pos_gt: Tensor,  
        atom_mask: Tensor  
    ) -> Tensor:  
        # Compute the distance maps for denoised and ground truth positions  
        denoised_distance_map = compute_distance_map(denoised_atom_pos, atom_mask)  
        gt_distance_map = compute_distance_map(atom_pos_gt, atom_mask)  
  
        # Compute the loss between the two distance maps  
        dist_map_loss = distance_map_loss(denoised_distance_map, gt_distance_map, atom_mask)  
  
        return dist_map_loss
    
    def forward(
        self,
        atom_inputs: Tensor,  
        atompair_inputs: Tensor,  
        atom_mask: Tensor,
        atom_pos: Tensor = None,
        return_loss: bool=True,
        **kwargs,
    ):

        atom_inputs = self.atom_inputs_to_atom_feats(atom_inputs)

        # embed inputs
        (
            atom_feats,
            atompair_feats,
        ) = self.input_embedder(
            atom_inputs=atom_inputs,
            atompair_inputs=atompair_inputs,
        )

        
        if not return_loss:
            # sampling
            denoised_atom_pos = self.diffuser.sample(
                atom_mask=atom_mask,
                num_sample_steps=self.num_sample_steps,
                return_all_timesteps=False,
                atom_feats=atom_feats,
                atompair_feats=atompair_feats,
            )
        else:
            (
                diffusion_loss,
                denoised_atom_pos,
                sigmas,
            ) = self.diffuser(
                atom_pos,
                atom_mask=atom_mask,
                atom_feats=atom_feats,
                atompair_feats=atompair_feats,
            )
        

        distance_map_loss = self.compute_loss_with_distance_map(denoised_atom_pos, atom_pos, atom_mask)
        
        out = ProteinDiffusionOutput(
            loss = diffusion_loss + self.distance_loss_weight * distance_map_loss if return_loss else None,
            denoised_atom_pos=denoised_atom_pos,
            sigmas=sigmas if return_loss else None,
            distance_map_loss=distance_map_loss,
            diffusion_loss=diffusion_loss if return_loss else None,
        )
        return out







import numpy as np
import matplotlib.pyplot as plt

def linear_schedule(num_steps, sigma_min, sigma_max, sigma_data, rho):
    """
    Computes the noise schedule as used in the diffusion protein design code.


    The idea is to first interpolate linearly in the space of (sigma^(1/rho))  
    then re-power by rho and scale by sigma_data. Finally, a final zero is appended.  
    
    Args:  
    num_steps (int): Number of steps (not counting the final zero step)  
    sigma_min (float): The minimum sigma value (before scaling) at the last step.  
    sigma_max (float): The maximum sigma value (before scaling) at the first step.  
    sigma_data (float): A scalar multiplier applied to all sigma values.  
    rho (float): Exponent controlling the shape of the schedule.  
        
    Returns:  
    A numpy array of length (num_steps+1) containing noise levels.  
    """  
    t = np.linspace(0, 1, num_steps)  # normalized time steps from 0 to 1  
    # Interpolate in the (sigma^(1/rho)) space.  
    sched = ( sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho)) )**rho  
    sched = sched * sigma_data  # apply scaling  
    # Append final sigma of 0 to mimic the code.  
    return np.concatenate([sched, np.array([0.0])])



def cosine_schedule(num_steps, sigma_min, sigma_max, sigma_data):
    """
    Computes a cosine-based noise schedule which goes from sigma_maxsigma_data at t=0
    to sigma_minsigma_data at t=1 and then appends a final step of 0.


    We use a simple cosine interpolation. Specifically, for t in [0,1]:  
    sigma(t) = sigma_min_eff + (sigma_max_eff - sigma_min_eff)*cos( (pi/2)*t )  
    where:  
    sigma_max_eff = sigma_max * sigma_data, and sigma_min_eff = sigma_min * sigma_data.  
    At t=0: cos(0) = 1 so sigma = sigma_max_eff.  
    At t=1: cos(pi/2)= 0 so sigma = sigma_min_eff.  
    Args:  
    num_steps (int): Number of steps (not counting the final 0 step)  
    sigma_min (float): The minimum sigma (before scaling)  
    sigma_max (float): The maximum sigma (before scaling)  
    sigma_data (float): Multiplicative scale.  
        
    Returns:  
    A numpy array of length (num_steps+1) containing noise levels.  
    """  
    sigma_min_eff = sigma_min * sigma_data  
    sigma_max_eff = sigma_max * sigma_data  
    t = np.linspace(0, 1, num_steps)  # normalized time steps  
    sched = sigma_min_eff + (sigma_max_eff - sigma_min_eff) * np.cos(t * np.pi/2)  
    return np.concatenate([sched, np.array([0.0])])  


def plot_noise_schedules(num_steps=32):
    """
    Plots the noise schedules from the linear-style (for several ρ values) and the cosine schedule.
    """
    # Parameters -- modify these to explore different regimes.
    sigma_min = 0.002
    sigma_max = 80
    sigma_data = 0.5


    # Choose several rho values to see its effect on the linear schedule shape.  
    rhos = [1, 3, 7, 10]  
    
    plt.figure(figsize=(10,6))  
    
    # Plot linear schedules for different ρ values.  
    for rho in rhos:  
        sched_linear = linear_schedule(num_steps, sigma_min, sigma_max, sigma_data, rho)  
        plt.plot(np.arange(len(sched_linear)), sched_linear, label=f'Linear (ρ={rho})')  
    
    # Plot the cosine schedule (using the same sigma_min and sigma_max endpoints).  
    sched_cosine = cosine_schedule(num_steps, sigma_min, sigma_max, sigma_data)  
    plt.plot(np.arange(len(sched_cosine)), sched_cosine, label='Cosine', linestyle='--', color='black')  
    
    plt.xlabel("Time step")  
    plt.ylabel("Sigma (noise level)")  
    plt.title("Comparison of Noise Schedules (Linear vs Cosine)")  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()
    plt.savefig("noise_schedules.png")

if __name__ == "__main__":
    plot_noise_schedules()