import torch
def normalize(v):
    return v / torch.norm(v)

