
import torch
import torch.nn as nn


def _normalization(kind, channels):
    if kind == "GN":
        return nn.GroupNorm(2, channels)
    if kind == "BN":
        return nn.BatchNorm2d(channels)
    raise ValueError("normalization must be GN or BN")


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        kind,
        groups=1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            _normalization(kind, out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, kind):
        super().__init__()
        hidden_channels = int(round(in_channels * expansion))
        layers = []
        if expansion != 1:
            layers.append(
                ConvNormActivation(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    stride=1,
                    kind=kind,
                )
            )
        layers.extend(
            [
                ConvNormActivation(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    kind=kind,
                    groups=hidden_channels,
                ),
                nn.Conv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                _normalization(kind, out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs):
        outputs = self.block(inputs)
        if self.use_residual:
            outputs = inputs + outputs
        return outputs


class MobileNetV2(nn.Module):

    configuration = (
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )

    def __init__(self, num_classes=10, normalization="GN"):
        super().__init__()
        input_channels = 32
        layers = [
            ConvNormActivation(
                3,
                input_channels,
                kernel_size=3,
                stride=1,
                kind=normalization,
            )
        ]
        for expansion, channels, repeats, first_stride in self.configuration:
            for index in range(repeats):
                stride = first_stride if index == 0 else 1
                layers.append(
                    InvertedResidual(
                        input_channels,
                        channels,
                        stride,
                        expansion,
                        normalization,
                    )
                )
                input_channels = channels
        layers.append(
            ConvNormActivation(
                input_channels,
                1280,
                kernel_size=1,
                stride=1,
                kind=normalization,
            )
        )
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = self.pool(outputs).flatten(1)
        return self.classifier(outputs)


def test():
    model = MobileNetV2(num_classes=100, normalization="GN")
    output = model(torch.randn(2, 3, 32, 32))
    print(output.shape)
    print(sum(parameter.numel() for parameter in model.parameters()))


if __name__ == "__main__":
    test()
