import torch
from typing import Tuple

def dynamic_distance_bias_matrix(start: Tuple, stop: Tuple )-> torch.Tensor:
    """ generate a specific subset of the full alibi matrix """
    res = -torch.abs(torch.arange(start[0], start[1]) - torch.arange(stop[0],stop[1])[:,None])
    start_height = start[1]-start[0]
    stop_width = stop[1]-stop[0]
    if res.shape[0] <= start_height:
        res = res.T
    assert res.shape[0] == start_height, f"mismatch in generated mask 0 dim, {res.shape[0]=}, {start_height=}"
    assert res.shape[1] == stop_width,f"mismatch in generated mask 1 dim, {res.shape[1]=}, {stop_width=}" 
    return res