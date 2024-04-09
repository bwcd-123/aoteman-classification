import torch
from torch import nn


class AddNoise(nn.Module):
    """
    Add noise to images.(recommand to input Tensor)

    If input is Tensor(float), std=0.01.

    If input is image(uint8), std=1.

    Parameters:

    """
    def __init__(self, image_noise_scale=1, tensor_noise_scale=0.01, *args, **kwargs) -> None:
        super(AddNoise, self).__init__(*args, **kwargs)
        self.image_noise_scale = image_noise_scale
        self.tensor_noise_scale = tensor_noise_scale


    def forward(self, x):
        noise = torch.normal(0., self.image_noise_scale, x.shape)
        print(noise)
        return x
        # if isinstance(x, torch.Tensor):
        #     noise = torch.normal(0., self.noise_scale, x.shape)
        # else:
        #     noise = torch.normal(0., self.noise_scale, x.zise)
        # return x + noise