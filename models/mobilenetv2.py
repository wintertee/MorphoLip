from typing import Callable, List, Literal, Optional

import torch
from torch import Tensor, nn

from .layers import BatchNorm2d, Conv1x1, DilationConv, InfinityConv


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        hidden_dim: int,
        norm_type: Literal["mean", "norm", "none", "normal"] = "norm",
        use_relu: bool = True,
        morpho_type: Literal["dilation", "infinity", "none", "maxpool"] = "none",
        conv_type: Literal["normal", "norm1", "norminf", "d_inf"] = "normal",
    ) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        self.use_res_connect = stride == 1 and in_dim == out_dim
        activation_layer = nn.ReLU if use_relu else nn.Identity

        # pw
        if in_dim != hidden_dim:
            self.ex = Conv1x1(in_dim, hidden_dim, stride=1, conv_type=conv_type)
            self.ex_norm = BatchNorm2d(hidden_dim, norm_type=norm_type)
            self.ex_activation = activation_layer()
        else:
            self.ex = nn.Identity()
            self.ex_norm = nn.Identity()
            self.ex_activation = nn.Identity()

        # dw
        padding = 1
        kernel_size = 3

        if morpho_type == "dilation":
            self.dw = DilationConv(hidden_dim, kernel_size, stride, padding)
            self.dw_activation = nn.Identity()

        elif morpho_type == "infinity":
            self.dw = InfinityConv(hidden_dim, kernel_size, stride, padding)
            self.dw_activation = nn.Identity()
        elif morpho_type == "none":
            self.dw = nn.Conv2d(
                hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim
            )
            self.dw_activation = nn.ReLU()
        else:
            raise ValueError("morpho should be one of 'dilation', 'infinity', 'none'")

        self.dw_norm = BatchNorm2d(hidden_dim, norm_type=norm_type)

        # pw
        self.pw = Conv1x1(hidden_dim, out_dim, stride=1, conv_type=conv_type)
        self.pw_norm = BatchNorm2d(out_dim, norm_type=norm_type)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            identity = x
        x = self.ex(x)
        x = self.ex_norm(x)
        x = self.ex_activation(x)
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
        morpho_type: Literal["dilation", "infinity", "none", "maxpool"] = "none",
        morpho_init: Literal["uniform", "normal"] = "normal",
        norm_type: Literal["mean", "norm", "none", "normal"] = "normal",
        dropout: float = 0.2,
        use_relu: bool = True,
        conv_type: Literal["normal", "norm1", "norminf", "d_inf"] = "normal",
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
                norm_type=norm_type,
                use_relu=use_relu,
                morpho_type=morpho_type,
                conv_type=conv_type,
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
                        norm_type=norm_type,
                        morpho_type=morpho_type,
                        conv_type=conv_type,
                        use_relu=use_relu,
                    )
                )
                input_channel = output_channel
        # building last several layers
        if input_channel != last_channel:
            features.append(
                nn.Sequential(
                    Conv1x1(input_channel, last_channel, stride=1, type=conv_type),
                    BatchNorm2d(last_channel, norm_type=norm_type),
                    nn.ReLU(),
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
