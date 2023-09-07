import torch
import torch.nn.functional as F
from torch import Tensor


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
