from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .morphology import dilation, infinity


class NormLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps  = 1e-5
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

        self.register_buffer("running_var", torch.ones(1))

        self.bias = nn.Parameter(torch.empty(out_dim))
        nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            weight_norm = torch.linalg.vector_norm(
                self.weight.data, ord=1, dim=1, keepdim=True
            )
            self.weight.data.div_(weight_norm.clamp(min=1.0))

        x = F.linear(x, self.weight, self.bias)

        if self.training:
            var = x.var().float()
            # Update running mean using momentum=0.1
            with torch.no_grad():
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            var = self.running_var

        x = x / torch.sqrt(var + self.eps) + self.bias[None, :]

        return x


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
        nn.init.uniform_(self.se, a=-1, b=1)

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
        out = infinity(x, self.se, self.stride, self.padding)
        # print("inf", out.var(dim=(0, 2, 3)).mean().item())

        return out


class Conv1x1(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        conv_type: Literal["normal", "norm1", "norminf", "d_inf","halfnorm"] = "normal",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.conv_type = conv_type

        self.weight = nn.Parameter(
            torch.empty(out_dim, in_dim, 1, 1), requires_grad=True
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        # print(x.var(dim=(0, 2, 3)).mean().item())
        if self.conv_type == "normal":
            return F.conv2d(x, self.weight, stride=self.stride)
        elif self.conv_type == "norm1":
            self.constrain_norm1()
            return F.conv2d(x, self.weight, stride=self.stride)
        elif self.conv_type == "norminf":
            self.constrain_norminf()
            return F.conv2d(x, self.weight, stride=self.stride)
        elif self.conv_type == "d_inf":
            # TODO: CUDA out of memory
            if self.stride != 1:
                x = x[:, :, :: self.stride, :: self.stride]

            x = x.unsqueeze(1)  # shape: [N, 1, C, H, W]
            weight = self.weight.unsqueeze(0)  # shape: [1, out_dim, C, 1, 1]

            x = torch.abs(x + weight)
            x, _ = torch.max(x, dim=2)
            return x
        elif self.conv_type == "halfnorm":
            with torch.no_grad():
                new_weight = self.weight.data.squeeze()
                new_weight = torch.sqrt(self.abs(new_weight))
                new_weight = torch.sum(new_weight, dim=1)
                halfnorm = new_weight.max()
            x = F.conv2d(x, self.weight, stride = self.stride)
            x = x / halfnorm;            

    @torch.no_grad()
    def constrain_norm1(self) -> None:
        norm1 = torch.linalg.vector_norm(
            self.weight.data, ord=1, dim=1, keepdim=True
        ).clamp(min=1)
        self.weight.data.div_(norm1)

    @torch.no_grad()
    def constrain_norminf(self) -> None:
        size = self.weight.data.size(1)
        norminf = torch.linalg.vector_norm(
            self.weight.data, ord=float("inf"), dim=1, keepdim=True
        )
        self.weight.data = self.weight.data.div_((norminf * size).clamp(min=1))


class MeanBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, input):
        with torch.no_grad():
            if self.training:
                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3])
                # Update running mean using momentum=0.1
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
            else:
                mean = self.running_mean
            # print(self.training, mean.mean().item())

        return input - mean[None, :, None, None] + self.bias[None, :, None, None]


class NormBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int):
        super().__init__(num_features)

    def forward(self, input):
        # Apply the constraint on the weights first
        with torch.no_grad():
            norm_inf = torch.abs(self.weight / (self.running_var + self.eps)).clamp(
                min=1.0
            )
            self.weight.data.div_(norm_inf)
        # Then, apply the original batch normalization
        output = super().forward(input)

        return output


class GlobalNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(1))

    def forward(self, x: torch.Tensor):
        # Apply the constraint on the weights first

        if self.training:
            mean = x.mean([0, 2, 3]).float()
            var = x.var().float()
            # Update running mean using momentum=0.1
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

            self.weight.data.clamp_(max=1.0)
        normalized_weight = self.weight / torch.sqrt(var + self.eps)
        x = (
            x * normalized_weight[None, :, None, None]
            - mean[None, :, None, None] * normalized_weight[None, :, None, None]
            + self.bias[None, :, None, None]
        )

        return x


class BatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        norm_type: Literal["mean", "norm", "none", "normal", "globalnorm"],
    ):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "mean":
            self.bn = MeanBatchNorm2d(num_features)
        elif norm_type == "norm":
            self.bn = NormBatchNorm2d(num_features)
        elif norm_type == "none":
            self.bn = nn.Identity()
        elif norm_type == "normal":
            self.bn = nn.BatchNorm2d(num_features)
        elif norm_type == "globalnorm":
            self.bn = GlobalNorm2d(num_features)
        else:
            raise ValueError(
                f"norm_type should be one of 'mean', 'norm', 'none', 'normal', 'globalnorm'"
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x)

    def extra_repr(self) -> str:
        return f"norm_type={self.norm_type}"
