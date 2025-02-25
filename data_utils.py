import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
import pandas as pd
from lightning import LightningDataModule
import functools

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
# For a 1D tensor x, returns a tensor y with:
# y[0] = 0 and y[i] = sum(x[:i]) for i > 0.
    return torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), torch.cumsum(x, dim=0)[:-1]))

def make_np_example(coords_dict):
    """Make a dictionary of non-batched numpy protein features."""
    bb_atom_types = ['N', 'CA', 'C', 'O']
    bb_idx = [i for i, atom_type in enumerate(atom_types)
              if atom_type in bb_atom_types]

    num_res = np.array(coords_dict['N']).shape[0]
    atom_positions = np.zeros([num_res, 37, 3], dtype=float)

    for i, atom_type in enumerate(atom_types):
        if atom_type in bb_atom_types:
            atom_positions[:, i, :] = np.array(coords_dict[atom_type])

    # Mask nan / None coordinates.
    nan_pos = np.isnan(atom_positions)[..., 0]
    atom_positions[nan_pos] = 0.
    atom_mask = np.zeros([num_res, 37])
    atom_mask[..., bb_idx] = 1
    atom_mask[nan_pos] = 0

    batch = {
        'atom_positions': atom_positions,
        'atom_mask': atom_mask,
        'residue_index': np.arange(num_res)
    }
    return batch

def center_positions(np_example):
  """Center 'atom_positions' on CA center of mass."""
  atom_positions = np_example['atom_positions']
  atom_mask = np_example['atom_mask']
  ca_positions = atom_positions[:, 1, :]
  ca_mask = atom_mask[:, 1]

  ca_center = (np.sum(ca_mask[..., None] * ca_positions, axis=0) /
   (np.sum(ca_mask, axis=0) + 1e-9))
  atom_positions = ((atom_positions - ca_center[None, ...]) *
                    atom_mask[..., None])
  np_example['atom_positions'] = atom_positions

def make_fixed_size(np_example, max_seq_length=500):
    """Pad features to fixed sequence length, i.e. currently axis=0."""
    for k, v in np_example.items():
        pad = max_seq_length - v.shape[0]
        if pad > 0:
            v = np.pad(v, ((0, pad),) + ((0, 0),) * (len(v.shape) - 1))
        elif pad < 0:
            v = v[:max_seq_length]
        np_example[k] = v

def make_fixed_size_seq(np_example, max_seq_length=500, pad_token="<PAD>"):
    """
    Pads or truncates the sequence in np_example to a fixed length.
    The sequence in np_example["seq"] is expected to be a string or a list of characters.
    """
    seq = list(np_example["seq"])
    seq_len = len(seq)


    if seq_len < max_seq_length:  
        # Pad the sequence with the pad token  
        seq = seq + [pad_token] * (max_seq_length - seq_len)  
    elif seq_len > max_seq_length:  
        # Truncate sequence to max_seq_length  
        seq = seq[:max_seq_length]  
        
    np_example["seq"] = seq  

class DatasetFromDataframe(Dataset):
    def __init__(self, data_frame, max_seq_length=512):
        self.data = data_frame
        self.max_seq_length = max_seq_length
        amino_acid_vocab = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYVX")} 
        amino_acid_vocab.update({"<PAD>": len(amino_acid_vocab)})
        self.amino_acid_vocab = amino_acid_vocab
        self.vocab_size = len(amino_acid_vocab)
        self.padding_idx = self.amino_acid_vocab["<PAD>"]
        self.pad = "<PAD>"

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, idx):  
        # Assume that each dataframe row has an attribute 'coords'  
        # which is processed by the provided functions.  
        coords_dict = self.data.iloc[idx].coords  
        np_example = make_np_example(coords_dict)      # returns a dict with keys including "atom_positions" and "atom_mask"  

        make_fixed_size(np_example, self.max_seq_length)  # pads or truncates to max_seq_length  
        center_positions(np_example)                    # centers positions on the center of mass 
        seq = self.data.iloc[idx].seq
        np_example.update({"seq": seq})
        make_fixed_size_seq(np_example, self.max_seq_length, self.pad)  # pads or truncates the sequence to max_seq_length
        # center seq
        # Restrict to backbone atoms.  
        # Original "atom_positions" has shape [max_seq_length, 37, 3].  
        # We want only indices 0, 1, 2 and 4 (i.e. 4 atomic positions).  
        backbone_indices = [0, 1, 2, 4]  
        np_example["atom_positions_reduced"] = np_example["atom_positions"][:, backbone_indices, :]  
        np_example["atom_mask_reduced"] = np_example["atom_mask"][:, backbone_indices]
        # get the sequence in np_example["seq"] and convert it to a list of integers using the amino_acid_vocab
        np_example["seq"] = np.array([self.amino_acid_vocab[aa] for aa in np_example["seq"]])
 

        # Convert numpy arrays to torch tensors.  
        example = {k: torch.tensor(v, dtype=torch.float32) for k, v in np_example.items()}  
        return example  

def get_split(pdb_name, cath_splits):
    if pdb_name in cath_splits.train[0]:
        return 'train'
    elif pdb_name in cath_splits.validation[0]:
        return 'validation'
    elif pdb_name in cath_splits.test[0]:
        return 'test'
    else:
        return 'None'

def atom_ref_pos_to_atompair_inputs(
    atom_ref_pos: Tensor # shape can be [m, 3] or [B, m, 3]
    ) -> Tensor:
    """
    Compute atompair inputs from atom reference positions.


    The function computes:  
    1. Pairwise relative positions.  
    2. Inverse squared distances.  
    3. A mask (here just ones).  
        
    Then it concatenates these features along the last dimension.  
    
    :param atom_ref_pos: Tensor of shape [B, m, 3] or [m, 3]  
    :return: Tensor of shape [B, m, m, 5] or [m, m, 5] (if input was not batched).  
            The five features along the last dim are:  
                - Relative positions (3 values).  
                - Inverse squared distance (1 value).  
                - Mask (1 value).  
    """  
    # If the input is not batched, add a batch dimension.  
    added_batch = False  
    if atom_ref_pos.dim() == 2:  
        atom_ref_pos = atom_ref_pos.unsqueeze(0)   # now shape: [1, m, 3]  
        added_batch = True  

    # Assume atom_ref_pos is now of shape [B, m, 3]  
    B, m, _ = atom_ref_pos.shape  

    # Compute pairwise relative positions using broadcasting:  
    # Shape: [B, m, m, 3]  
    pairwise_rel_pos = atom_ref_pos.unsqueeze(2) - atom_ref_pos.unsqueeze(1)  

    # Compute the inverse squared distances:  
    # Here we compute the norm over the last dimension and then the inverse of (1 + squared norm)  
    # Shape: [B, m, m]  
    distances = pairwise_rel_pos.norm(p=2, dim=-1)  
    atom_inv_square_dist = (1.0 + distances ** 2).reciprocal()  

    # Create a mask. Here we simply use ones to indicate all atoms come from the same reference space.  
    # Shape: [B, m, m]  
    same_ref_space_mask = torch.ones_like(atom_inv_square_dist, dtype=torch.bool)  

    # Pack the features into a single tensor.  
    # We have:  
    #   • pairwise_rel_pos: 3 channels,  
    #   • atom_inv_square_dist: 1 channel (unsqueezed to shape [B, m, m, 1])  
    #   • same_ref_space_mask (converted to float, unsqueezed to shape [B, m, m, 1])  
    # and then concatenate along the channel dimension.  
    atompair_inputs = torch.cat(  
        [  
            pairwise_rel_pos,  # [B, m, m, 3]  
            atom_inv_square_dist.unsqueeze(-1),  # [B, m, m, 1]  
            same_ref_space_mask.float().unsqueeze(-1)  # [B, m, m, 1]  
        ],  
        dim=-1  # Now the last dimension becomes 5.  
    )  # Final shape: [B, m, m, 5]  

    # If we added a dummy batch, remove it before returning.  
    if added_batch:  
        atompair_inputs = atompair_inputs.squeeze(0)  

    return atompair_inputs  

def protein_to_atom_collate_fn(samples, dim_atom_inputs: int = 3, default_num_molecule_mods: int = 5):

    # "atom_positions": shape [B, seq_len, atoms_per_window, 3],
    # "atom_mask": shape [B, seq_len, atoms_per_window] etc.
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0].keys()}
   
    B, seq_len, atoms_per_window, _ = batch["atom_positions_reduced"].shape  
    atom_seq_len = seq_len * atoms_per_window  # total atom slots per protein  
    atom_offsets = batch["residue_index"]  # [B, seq_len]

    # Compute per-residue valid atom counts.  
    # "atom_mask" has shape [B, seq_len, atoms_per_window] and should be 1 when an atom is present.  
    molecule_atom_lens = batch["atom_mask_reduced"].sum(dim=-1).long()  # # not sure this is correct shape: [B, seq_len]  

    # Flatten atom positions: reshape from [B, seq_len, atoms_per_window, 3] to [B, atom_seq_len, 3]  
    flat_atom_pos = batch["atom_positions_reduced"].reshape(B, atom_seq_len, 3)  
    flat_atom_mask = batch["atom_mask_reduced"].reshape(B, atom_seq_len).bool()  # [B, atom_seq_len]
    
    # atom_inputs = batch["seq"].repeat_interleave(atoms_per_window, dim=1).long()  # [B, atom_seq_len]

    atom_inputs = flat_atom_mask.clone().float()  # [B, atom_seq_len]
 
    # atom_inputs = torch.zeros(B, atom_seq_len, dim_atom_inputs, dtype=torch.float32)  
    # atom_inputs[:, :, :3] = flat_atom_pos  

    # Create atompair_inputs as a dummy tensor (shape: [B, atom_seq_len, atom_seq_len, 5]).  
    atompair_inputs = atom_ref_pos_to_atompair_inputs(flat_atom_pos)  # [B, atom_seq_len, atom_seq_len, 5]
 
    # Pack all outputs into a dictionary in the expected format.  
    out = {  
        "atom_inputs": atom_inputs,                       # [B, atom_seq_len, dim_atom_inputs]  
        "atompair_inputs": atompair_inputs,               # [B, atom_seq_len, atom_seq_len, 5]  
        "atom_pos": flat_atom_pos,                        # [B, atom_seq_len, 3] 
        "atom_mask": flat_atom_mask,                # [B, atom_seq_len] 
        "molecule_atom_lens": molecule_atom_lens,         # [B, seq_len]    
        "atom_offsets": atom_offsets,                      # [B, seq_len]  
    }  
    return out    

class ProteinDataModule(LightningDataModule):
    def __init__(self, batch_size=64, batch_size_eval=4, max_seq_length=256, num_workers=24, dim_atom_inputs_init=3, default_num_molecule_mods=5):
        super().__init__()
        self.collate_fn = functools.partial(  
        protein_to_atom_collate_fn,  
        dim_atom_inputs=dim_atom_inputs_init,
        default_num_molecule_mods=default_num_molecule_mods  
        )
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        # Datasets will be created in setup  
        self.train_dataset = None  
        self.val_dataset = None  
        self.test_dataset = None 

    def prepare_data(self):  
        print('Reading chain_set.jsonl, this can take 1 or 2 minutes...')
        df = pd.read_json('/home/t-brahmani/babak/alphafold3-pytorch-lightning-hydra/my_data/chain_set.jsonl', lines=True)
        cath_splits = pd.read_json('/home/t-brahmani/babak/alphafold3-pytorch-lightning-hydra/my_data/chain_set_splits.json', lines=True)
        print('Read data.')

        df['split'] = df.name.apply(lambda x: get_split(x, cath_splits))
        df['seq_len'] = df.seq.apply(lambda x: len(x)) 
        self.train_df = df[df.split == 'train']
        self.val_df = df[df.split == 'validation']
        self.test_df = df[df.split == 'test']

    def setup(self, stage=None):  
        if stage == "fit" or stage is None:  
            self.train_dataset = DatasetFromDataframe(self.train_df, max_seq_length=self.max_seq_length)  
            self.val_dataset = DatasetFromDataframe(self.val_df, max_seq_length=self.max_seq_length)
        
        if stage == "test" or stage is None:  
            self.test_dataset = DatasetFromDataframe(self.test_df, max_seq_length=self.max_seq_length) 

    def train_dataloader(self):  
        return DataLoader(  
            self.train_dataset,  
            batch_size=self.batch_size,  
            shuffle=True,  
            collate_fn=self.collate_fn,  
            num_workers=self.num_workers,  
            pin_memory=True,
            drop_last=True 
        )  

    def val_dataloader(self):  
        return DataLoader(  
            self.val_dataset,  
            batch_size=self.batch_size_eval,
            shuffle=False,  
            collate_fn=self.collate_fn,  
            num_workers=self.num_workers,  
            pin_memory=True,
            drop_last=True 
        )  

    def test_dataloader(self):  
        return DataLoader(  
            self.test_dataset,  
            batch_size=self.batch_size_eval,
            shuffle=False,  
            collate_fn=self.collate_fn,  
            num_workers=self.num_workers,  
            pin_memory=True  
        )  

if __name__ == "__main__":

    print('Reading chain_set.jsonl, this can take 1 or 2 minutes...')
    df = pd.read_json('/home/t-brahmani/babak/alphafold3-pytorch-lightning-hydra/my_data/chain_set.jsonl', lines=True)
    cath_splits = pd.read_json('/home/t-brahmani/babak/alphafold3-pytorch-lightning-hydra/my_data/chain_set_splits.json', lines=True)
    print('Read data.')

    df['split'] = df.name.apply(lambda x: get_split(x, cath_splits))
    df['seq_len'] = df.seq.apply(lambda x: len(x))

    train_protein_dataset = DatasetFromDataframe(df[df.split == 'train'], max_seq_length=256)
    train_loader = DataLoader(train_protein_dataset, batch_size=1, collate_fn=protein_to_atom_collate_fn)
    batch = next(iter(train_loader))
    print("atom_inputs shape:", batch["atom_inputs"].shape) # Expected: [64, 25637, 77]
    print("atompair_inputs shape:", batch["atompair_inputs"].shape) # Expected: [64, 25637, 25637, 5]
    print("molecule_atom_lens shape:", batch["molecule_atom_lens"].shape) # Expected: [64, 256]
    print("atom_pos shape:", batch["atom_pos"].shape) # Expected: [64, 25637, 3]

    # Compute sigma_data across the entire training set.  
    total_sq_sum = 0.0  
    total_count = 0  
    for batch in train_loader:  
        # batch["atom_pos"]: tensor of shape [B, atom_seq_len, 3]  
        # batch["atom_mask"]: tensor of shape [B, atom_seq_len] (converted to bool in protein_to_atom_collate_fn)  
        atom_pos = batch["atom_pos"]  
        atom_mask = batch["atom_mask"]  
        
        # Select only valid atom positions using the mask.  
        valid_positions = atom_pos[atom_mask]  
        total_sq_sum += (valid_positions ** 2).sum().item()  
        total_count += valid_positions.numel()  # counts all coordinate values  

    sigma_data = np.sqrt(total_sq_sum / total_count)  
    print("Computed sigma_data:", sigma_data)
    #visialaize 2 examples of atom_mask and save the figure



