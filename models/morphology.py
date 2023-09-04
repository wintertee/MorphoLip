import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
def dilation(
    input: Tensor,
    se: Tensor,
    stride: int = 1,
    padding: int = 1,
    cval: float = -1e4,
) -> Tensor:
    c, se_h, se_w = se.shape
    output: Tensor = F.pad(
        input, (padding, padding, padding, padding), mode="constant", value=cval
    )
    output = output.unfold(2, se_h, stride).unfold(3, se_w, stride)
    output = output + se.view(1, c, 1, 1, se_h, se_w)
    output, _ = torch.max(output, 4)
    output, _ = torch.max(output, 4)
    # output = torch.amax(output, (4, 5))
    return output


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
def infinity(
    tensor: Tensor,
    se: Tensor,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    c, se_h, se_w = se.shape
    output: Tensor = F.pad(
        tensor, (padding, padding, padding, padding), mode="constant", value=0
    )
    output = output.unfold(2, se_h, stride).unfold(3, se_w, stride)
    output = output + se.view(1, c, 1, 1, se_h, se_w)
    output = torch.abs(output)
    output, _ = torch.max(output, 4)
    output, _ = torch.max(output, 4)
    return output


class DilationConv(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.se = nn.Parameter(
            torch.empty(self.channels, self.kernel_size, self.kernel_size).squeeze()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dilation(x, self.se, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}"
        )


class InfinityConv(DilationConv):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return infinity(x, self.se, self.stride, self.padding)
