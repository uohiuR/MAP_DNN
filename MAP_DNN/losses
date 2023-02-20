import torch

def loss_fn(out, post, pre):
    R = out / ratio
    z = 2 * (torch.sqrt(post + 3 / 8) - torch.sqrt((pre + 3 / 8) * R))
    z = z / torch.sqrt(1 + R)
    loss_z = torch.pow(z, 2) / 2
    loss_l1 = L1_rate * torch.abs(out)
    loss = loss_z + loss_l1
    return loss
