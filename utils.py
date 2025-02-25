

from typing import Tuple, Any
import torch
import os

from torch.nn import functional as F
from einops import rearrange
import einx
from einops import pack, rearrange, repeat, unpack
from torch import Tensor
import pandas as pd  
import matplotlib.pyplot as plt  


def identity(x, *args, **kwargs):
    """Return the input value."""
    return x

def log(t: Tensor, eps=1e-20) -> Tensor:
    """Run a safe log function that clamps the input to be above `eps` to avoid `log(0)`.

    :param t: The input tensor.
    :param eps: The epsilon value.
    :return: Tensor in the log domain.
    """
    return torch.log(t.clamp(min=eps))

def slice_at_dim(t: Tensor, dim_slice: slice, *, dim: int) -> Tensor:
    """Slice a Tensor at a specific dimension.

    :param t: The Tensor.
    :param dim_slice: The slice object.
    :param dim: The dimension to slice.
    :return: The sliced Tensor.
    """
    dim += t.ndim if dim < 0 else 0
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def pad_or_slice_to(t: Tensor, length: int, *, dim: int, pad_value=0) -> Tensor:
    """Pad or slice a Tensor to a specific length at a specific dimension.

    :param t: The Tensor.
    :param length: The length to pad or slice to.
    :param dim: The dimension to pad or slice.
    :param pad_value: The value to pad with.
    :return: The padded or sliced Tensor.
    """
    curr_length = t.shape[dim]

    if curr_length < length:
        t = pad_to_length(t, length, dim=dim, value=pad_value)
    elif curr_length > length:
        t = slice_at_dim(t, slice(0, length), dim=dim)

    return t

def pad_at_dim(t, pad: Tuple[int, int], *, dim=-1, value=0.0) -> Tensor:
    """Pad a Tensor at a specific dimension.

    :param t: The Tensor.
    :param pad: The padding.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)

def pad_to_length(t: Tensor, length: int, *, dim: int = -1, value=0) -> Tensor:
    """Pad a Tensor to a specific length at a specific dimension.

    :param t: The Tensor.
    :param length: The length to pad to.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    padding = max(length - t.shape[dim], 0)

    if padding == 0:
        return t

    return pad_at_dim(t, (0, padding), dim=dim, value=value)

def lens_to_mask(
    lens = None  # type: ignore
):
    """Convert a Tensor of lengths to a mask Tensor.

    :param lens: The lengths Tensor.
    :param max_len: The maximum length.
    :return: The mask Tensor.
    """
    device = lens.device
    if not_exists(max_len):
        max_len = lens.amax()
    arange = torch.arange(max_len, device=device)
    return einx.less("m, ... -> ... m", arange, lens)

def exclusive_cumsum(t: Tensor, dim: int = -1) -> Tensor:
    """Perform an exclusive cumulative summation on a Tensor.

    :param t: The Tensor.
    :param dim: The dimension to sum over.
    :return: The exclusive cumulative sum Tensor.
    """
    return t.cumsum(dim=dim) - t

def pack_one(t: Tensor, pattern: str):
    """Pack a single tensor into a tuple of tensors with the given pattern.

    :param t: The tensor to pack.
    :param pattern: The pattern with which to pack.
    :return: The packed tensor and the unpack function.
    """
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        """Unpack a single tensor.

        :param to_unpack: The tensor to unpack.
        :param pattern: The pattern with which to unpack.
        :return: The unpacked tensor.
        """
        (unpacked,) = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def batch_repeat_interleave(
    feat,  # type: ignore
    lens,  # type: ignore
    output_padding_value: (
        float | int | bool | None
    ) = None,  # NOTE: this value determines what the output padding value will be
):  # type: ignore
    """Batch repeat and interleave a sequence of features.

    :param feats: The features tensor.
    :param lens: The lengths tensor.
    :param output_padding_value: The output padding value.
    :return: The batch repeated and interleaved features tensor.
    """
    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device=device)

    offsets = exclusive_cumsum(lens)
    indices = einx.add("w, b n -> b n w", arange, offsets)

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.clamp(min=0).sum(dim=-1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device=device, dtype=torch.long)

    indices = indices.masked_fill(~mask, max_len)  # scatter to sink position for padding
    indices = rearrange(indices, "b n w -> b (n w)")

    # scatter

    seq_arange = torch.arange(seq, device=device)
    seq_arange = repeat(seq_arange, "n -> b (n w)", b=batch, w=window_size)

    # output_indices = einx.set_at('b [m], b nw, b nw -> b [m]', output_indices, indices, seq_arange)

    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    # output = einx.get_at('b [n] ..., b m -> b m ...', feats, output_indices)

    feats, unpack_one = pack_one(feats, "b n *")
    output_indices = repeat(output_indices, "b m -> b m d", d=feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    # set output padding value

    output_padding_value = default(output_padding_value, False if dtype == torch.bool else 0)

    output = einx.where("b n, b n ..., -> b n ...", output_mask, output, output_padding_value)

    return output



def to_pairwise_mask(
    mask_i,  # type: ignore
    mask_j = None,  # type: ignore
):  # type: ignore
    """Convert two masks into a pairwise mask.

    :param mask_i: The first mask.
    :param mask_j: The second mask.
    :return: The pairwise mask.
    """
    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and("... i, ... j -> ... i j", mask_i, mask_j)


def exists(val: Any) -> bool:
    """Check if a value exists.

    :param val: The value to check.
    :return: `True` if the value exists, otherwise `False`.
    """
    return val is not None

def not_exists(val: Any) -> bool:
    """Check if a value does not exist.

    :param val: The value to check.
    :return: `True` if the value does not exist, otherwise `False`.
    """
    return val is None

def default(v: Any, d: Any) -> Any:
    """Return default value `d` if `v` does not exist (i.e., is `None`).

    :param v: The value to check.
    :param d: The default value to return if `v` does not exist.
    :return: The value `v` if it exists, otherwise the default value `d`.
    """
    return v if exists(v) else d


def pad_to_multiple(t: Tensor, multiple: int, *, dim=-1, value=0.0) -> Tensor:
    """Pad a Tensor to a multiple of a specific number at a specific dimension.

    :param t: The Tensor.
    :param multiple: The multiple to pad to.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    seq_len = t.shape[dim]
    padding_needed = (multiple - (seq_len % multiple)) % multiple

    if padding_needed == 0:
        return t

    return pad_at_dim(t, (0, padding_needed), dim=dim, value=value)


def pad_and_window(t, window_size: int) -> Tensor:  # type: ignore
    """Pad and window a Tensor.

    :param t: The Tensor.
    :param window_size: The window size.
    :return: The padded and windowed Tensor.
    """
    t = pad_to_multiple(t, window_size, dim=1)
    t = rearrange(t, "b (n w) ... -> b n w ...", w=window_size)
    return t


def mean_pool_with_lens(
    feats,  # type: ignore
    lens,  # type: ignore
):  # type: ignore
    """Perform mean pooling on a Tensor with the given lengths.

    :param feats: The features Tensor.
    :param lens: The lengths Tensor.
    :return: The mean pooled Tensor.
    """
    seq_len = feats.shape[1]

    mask = lens > 0
    assert (
        lens.sum(dim=-1) <= seq_len
    ).all(), (
        "One of the lengths given exceeds the total sequence length of the features passed in."
    )

    cumsum_feats = feats.cumsum(dim=1)
    cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value=0.0)

    cumsum_indices = lens.cumsum(dim=1)
    cumsum_indices = F.pad(cumsum_indices, (1, 0), value=0)

    # sel_cumsum = einx.get_at('b [m] d, b n -> b n d', cumsum_feats, cumsum_indices)

    cumsum_indices = repeat(cumsum_indices, "b n -> b n d", d=cumsum_feats.shape[-1])
    sel_cumsum = cumsum_feats.gather(-2, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    avg = einx.divide("b n d, b n", summed, lens.clamp(min=1))
    avg = einx.where("b n, b n d, -> b n d", mask, avg, 0.0)
    return avg

def slice_at_dim(t: Tensor, dim_slice: slice, *, dim: int) -> Tensor:
    """Slice a Tensor at a specific dimension.

    :param t: The Tensor.
    :param dim_slice: The slice object.
    :param dim: The dimension to slice.
    :return: The sliced Tensor.
    """
    dim += t.ndim if dim < 0 else 0
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def pad_at_dim(t, pad: Tuple[int, int], *, dim=-1, value=0.0) -> Tensor:
    """Pad a Tensor at a specific dimension.

    :param t: The Tensor.
    :param pad: The padding.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)

def concat_previous_window(t: Tensor, *, dim_seq: int, dim_window: int) -> Tensor:
    """Concatenate the previous window of a Tensor.

    :param t: The Tensor.
    :param dim_seq: The sequence dimension.
    :param dim_window: The window dimension.
    :return: The concatenated Tensor.
    """
    t = pad_at_dim(t, (1, 0), dim=dim_seq, value=0.0)

    t = torch.cat(
        (
            slice_at_dim(t, slice(None, -1), dim=dim_seq),
            slice_at_dim(t, slice(1, None), dim=dim_seq),
        ),
        dim=dim_window,
    )

    return t

def full_pairwise_repr_to_windowed(
    pairwise_repr, window_size: int  # type: ignore
):  # type: ignore
    """Convert a full pairwise representation matrix to a local windowed one.

    :param pairwise_repr: The full pairwise representation matrix.
    :param window_size: The window size.
    :return: The local windowed pairwise representation matrix.
    """
    seq_len, device = pairwise_repr.shape[-2], pairwise_repr.device

    padding_needed = (window_size - (seq_len % window_size)) % window_size
    pairwise_repr = F.pad(pairwise_repr, (0, 0, 0, padding_needed, 0, padding_needed), value=0.0)
    pairwise_repr = rearrange(
        pairwise_repr, "... (i w1) (j w2) d -> ... i j w1 w2 d", w1=window_size, w2=window_size
    )
    pairwise_repr = concat_previous_window(pairwise_repr, dim_seq=-4, dim_window=-2)

    # get the diagonal

    n = torch.arange(pairwise_repr.shape[-4], device=device)

    # pairwise_repr = einx.get_at('... [i j] w1 w2 d, n, n -> ... n w1 w2 d', pairwise_repr, n, n)

    pairwise_repr = pairwise_repr[..., n, n, :, :, :]

    return pairwise_repr

def max_neg_value(t: Tensor) -> float:
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max

def full_attn_bias_to_windowed(
    attn_bias, window_size: int  # type: ignore
):  # type: ignore
    """Convert a full attention bias matrix to a local windowed one.

    :param attn_bias: The full attention bias matrix.
    :param window_size: The window size.
    :return: The local windowed attention bias matrix.
    """
    attn_bias = rearrange(attn_bias, "... -> ... 1")
    attn_bias = full_pairwise_repr_to_windowed(attn_bias, window_size=window_size)
    return rearrange(attn_bias, "... 1 -> ...")


def softclamp(t: Tensor, value: float) -> Tensor:
    """Perform a soft clamp on a Tensor.

    :param t: The Tensor.
    :param value: The value to clamp to.
    :return: The soft clamped Tensor
    """
    return (t / value).tanh() * value



def load_model_from_checkpoint(model,  ckpt_path):
    assert ckpt_path, "load_checkpoint must be provided for finetuning or eval"
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model from {ckpt_path} and epoch {ckpt['epoch']} and step {ckpt['global_step']}")
    return model



  
def plot_all_metrics(csv_file, save_dir): 
    # get the csv_file path from the logger and version of the trainer
    df = pd.read_csv(csv_file)
    plt.rcParams.update({'font.size': 14})
    df = df.fillna(method='ffill')  
    step = df['step'] 
       
    metric_columns = [col for col in df.columns if col not in ('epoch', 'step')]  
    import tqdm
    for metric in tqdm.tqdm(metric_columns, desc="Plotting metrics", unit="metric"):  
        plt.figure()  
        plt.plot(step, df[metric], ".", label=metric.replace('_', ' ').title() )
        plt.xlabel('Step')  
        plt.ylabel(metric.split('/')[-1])   
        plt.title(f'{metric.replace("_", " ").title()} Over Time')  
        plt.legend()  
        plt.show() 
        # rep;ace / with _ in metric
        metric = metric.replace('/', '_')
        plt.savefig(f"{save_dir}/{metric}.png")
    
    
if __name__ == "__main__":
    csv_file = "/home/t-brahmani/babak/alphafold3-pytorch-lightning-hydra/2025-02-15_17-59-28/metrics.csv"
    save_dir = "/home/t-brahmani/babak/alphafold3-pytorch-lightning-hydra/2025-02-15_17-59-28"
    plot_all_metrics(csv_file, save_dir)