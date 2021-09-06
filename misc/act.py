import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLU(nn.ReLU):
    def __init__(self, **kwargs) -> None:
        super(ReLU, self).__init__(**kwargs)


class PReLU(nn.PReLU):
    def __init__(self, **kwargs) -> None:
        super(PReLU, self).__init__(**kwargs)


class LeakyReLU(nn.LeakyReLU):
    def __init__(self, **kwargs) -> None:
        super(LeakyReLU, self).__init__(**kwargs)


class CELU(nn.CELU):
    def __init__(self, **kwargs) -> None:
        super(CELU, self).__init__(**kwargs)


class SELU(nn.SELU):
    def __init__(self, **kwargs) -> None:
        super(SELU, self).__init__(**kwargs)


class GELU(nn.GELU):
    def __init__(self, **kwargs) -> None:
        super(GELU, self).__init__(**kwargs)


class Softplus(nn.Softplus):
    def __init__(self, **kwargs) -> None:
        super(Softplus, self).__init__(**kwargs)


class Swish(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(input)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.tanh(F.softplus(input))

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
