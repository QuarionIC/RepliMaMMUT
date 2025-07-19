"""
Inspired by positional_encoding in [TubeViT](https://github.com/daniel-code/TubeViT/blob/main/tubevit/positional_encoding.py).
Add documentation and example cases for understanding
"""

from typing import Tuple

import torch
import numpy as np


def get_3d_sincos_pos_embed(
    embed_dim: int, tube_shape: Tuple[int, int, int], stride: Tuple[int, int, int], offset: Tuple[int, int, int], kernel_size: Tuple[int, int, int], cls_token: bool = False
) -> torch.Tensor:
    """
    Get 3D sine-cosine positional embedding.
    Args:
        embed_dim: embedding dimension
        tube_shape: (t_size, grid_h_size, grid_w_size)
        kernel_size: (t,h,w)
        offset: (t,h,w)
        stride: (t,h,w)
        cls_token: bool, whether to contain CLS token
    Returns:
        (torch.Tensor): [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim]
        (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 3 * 2
    embed_dim_temporal = embed_dim // 3

    # spatial
    grid_h_size = tube_shape[1]
    grid_h = torch.arange(grid_h_size, dtype=torch.float)
    grid_h = grid_h * stride[1] + offset[1] + kernel_size[1] // 2

    grid_w_size = tube_shape[2]
    grid_w = torch.arange(tube_shape[2], dtype=torch.float)
    grid_w = grid_w * stride[2] + offset[2] + kernel_size[2] // 2
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    t_size = tube_shape[0]
    grid_t = torch.arange(t_size, dtype=torch.float)
    grid_t = grid_t * stride[0] + offset[0] + kernel_size[0] // 2
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    pos_embed_temporal = pos_embed_temporal[:, None, :]
    pos_embed_temporal = torch.repeat_interleave(pos_embed_temporal, grid_h_size * grid_w_size, dim=1)
    pos_embed_spatial = pos_embed_spatial[None, :, :]
    pos_embed_spatial = torch.repeat_interleave(pos_embed_spatial, t_size, dim=0)

    pos_embed = torch.cat([pos_embed_temporal, pos_embed_spatial], dim=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])

    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """
    Get 2D sine-cosine positional embedding.
    Args:
        embed_dim: embedding dimension.
        grid_size: int of the grid height and width
        cls_token: bool, whether to contain CLS token
    Returns:
        (torch.Tensor): [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float)
    grid_w = torch.arange(grid_size, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    Get 2D sine-cosine positional embedding from grid.
    Args:
        embed_dim: embedding dimension.
        grid: positions
    Returns:
        (torch.Tensor): [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Get 1D sine-cosine positional embedding.
    Args:
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
    Returns:
        (torch.Tensor): tensor of shape (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb



def calc_conv_shape(video_shape: Tuple[int, int, int, int], kernel_size: Tuple[int, int, int], stride: Tuple[int, int, int], offset: Tuple[int, int, int]) -> np.ndarray:
    '''
    Get the 3d convolution shape (tube_shape) based on the video dimensions, kernel, stride and offset
    Args:
        video_shape: CTHW
        kernel_size: (t,h,w)
        stride: (t,h,w)
        offset: (t,h,w)
    Returns:
        (np.ndarray): array of shape (t,h,w)
    '''
    video_shape = np.array(video_shape)
    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    offset = np.array(offset)
    output = np.floor(((video_shape[[1, 2, 3]] - offset - kernel_size) / stride) + 1).astype(int)
    return output



if __name__ == "__main__":
    embed_dim = 32     # sometimes call the hidden dim
    video_shape = (3,120,240,240)  #CTHW
    stride = (16,32,32)
    offset = (0,0,0)
    kernel_size = (8,8,8)
    tube_shape = tuple(calc_conv_shape(video_shape, kernel_size, stride, offset))
    print(f'{tube_shape=}')
    cls_token = False
    result = get_3d_sincos_pos_embed(embed_dim,tube_shape,stride,offset,kernel_size,cls_token)
    print(f'{result.shape=}')


    embed_dim = 32     # sometimes call the hidden dim
    video_shape = (3,120,240,240)  #CTHW
    stride = (6,32,32)
    offset = (4,8,8)
    kernel_size = (16,4,4)
    tube_shape = tuple(calc_conv_shape(video_shape, kernel_size, stride, offset))
    print(f'{tube_shape=}')
    cls_token = False
    result = get_3d_sincos_pos_embed(embed_dim,tube_shape,stride,offset,kernel_size,cls_token)
    print(f'{result.shape=}')