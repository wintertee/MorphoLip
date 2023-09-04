from typing import Callable, List, Literal, Optional

import torch
from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation

from .morphology import DilationConv, InfinityConv


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        hidden_dim: int,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        morpho: Literal["dilation", "infinity", "none", "maxpool"] = "none",
    ) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        self.use_res_connect = stride == 1 and in_dim == out_dim

        # pw
        if in_dim != hidden_dim:
            self.expension = Conv2dNormActivation(
                in_dim,
                hidden_dim,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        else:
            self.expension = nn.Identity()

        # dw
        padding = 1
        kernel_size = 3

        if morpho == "dilation":
            self.dw = DilationConv(hidden_dim, kernel_size, stride, padding)
            self.dw_activation = nn.Identity()

        elif morpho == "infinity":
            self.dw = InfinityConv(hidden_dim, kernel_size, stride, padding)
            self.dw_activation = nn.Identity()
        elif morpho == "none":
            self.dw = nn.Conv2d(
                hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim
            )
            self.dw_activation = nn.ReLU6()
        else:
            raise ValueError("morpho should be one of 'dilation', 'infinity', 'none'")

        self.dw_norm = norm_layer(hidden_dim)

        # pw
        self.pw = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False)
        self.pw_norm = norm_layer(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            identity = x
        x = self.expension(x)
        x = self.dw(x)
        x = self.dw_norm(x)
        x = self.dw_activation(x)
        x = self.pw(x)
        x = self.pw_norm(x)
        if self.use_res_connect:
            x += identity
        return x


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        first_layer_stride: int = 2,
        input_channel: Optional[int] = None,
        last_channel: Optional[int] = None,
        block: Optional[Callable[..., nn.Module]] = InvertedResidual,
        morpho: Literal["dilation", "infinity", "none", "maxpool"] = "none",
        morpho_init: Literal["uniform", "normal"] = "normal",
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        dropout: float = 0.2,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()

        assert first_layer_stride in [1, 2]

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        if input_channel is None:
            input_channel = 32
        if last_channel is None:
            last_channel = 1280
        # building first layer
        features: List[nn.Module] = [
            block(
                3,
                input_channel,
                stride=first_layer_stride,
                hidden_dim=input_channel,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                morpho=morpho,
            )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        hidden_dim=t * input_channel,
                        norm_layer=norm_layer,
                        morpho=morpho,
                    )
                )
                input_channel = output_channel
        # building last several layers
        if input_channel != last_channel:
            features.append(
                Conv2dNormActivation(
                    input_channel,
                    last_channel,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU,
                )
            )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, DilationConv) or isinstance(m, InfinityConv):
                if morpho_init == "uniform":
                    nn.init.uniform_(m.se, -1, 1)
                elif morpho_init == "normal":
                    nn.init.normal_(m.se, 0, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
