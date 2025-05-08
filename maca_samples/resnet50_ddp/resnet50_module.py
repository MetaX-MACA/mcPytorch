import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        is_conv_to_cpu: bool,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_conv_to_cpu = is_conv_to_cpu
        if self.downsample is not None and self.is_conv_to_cpu is True:
            self.downsample_conv = conv1x1(inplanes, planes * self.expansion, stride=self.stride)
            self.downsample_bn = norm_layer(planes * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.is_conv_to_cpu:
            x = x.to(device_cpu)
            self.conv1.weight.data = self.conv1.weight.data.to(device_cpu)
        out = self.conv1(x)
        if self.is_conv_to_cpu:
            x = x.to(device_gpu)
            out = out.to(device_gpu)
        out = self.bn1(out)
        out = self.relu(out)
        if self.is_conv_to_cpu:
            out = out.to(device_cpu)
            self.conv2.weight.data = self.conv2.weight.data.to(device_cpu)
        out = self.conv2(out)
        if self.is_conv_to_cpu:
            out = out.to(device_gpu)
        out = self.bn2(out)
        out = self.relu(out)
        if self.is_conv_to_cpu:
            out = out.to(device_cpu)
            self.conv3.weight.data = self.conv3.weight.data.to(device_cpu)
        out = self.conv3(out)
        if self.is_conv_to_cpu:
            out = out.to(device_gpu)
        out = self.bn3(out)
        if self.downsample is not None:
            if self.is_conv_to_cpu:
                x = x.to(device_cpu)
                self.downsample_conv.weight.data = self.downsample_conv.weight.data.to(device_cpu)
                identity = self.downsample_conv(x)
                x = x.to(device_gpu)
                identity = identity.to(device_gpu)
                identity = self.downsample_bn(identity)
            else:
                identity = self.downsample(x)
        out1 = identity + out
        out2 = self.relu(out1)
        return out2


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        is_conv_to_cpu: bool,
        is_check: bool,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.is_conv_to_cpu = is_conv_to_cpu
        self.is_check = is_check
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], is_conv_to_cpu=self.is_conv_to_cpu)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       is_conv_to_cpu=self.is_conv_to_cpu)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       is_conv_to_cpu=self.is_conv_to_cpu)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       is_conv_to_cpu=self.is_conv_to_cpu)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, is_conv_to_cpu: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.is_conv_to_cpu, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.is_conv_to_cpu, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        if self.is_conv_to_cpu:
            x = x.to(device_cpu)
            self.conv1.weight.data = self.conv1.weight.data.to(device_cpu)
        x = self.conv1(x)
        if self.is_conv_to_cpu:
            x = x.to(device_gpu)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[Bottleneck]],
    layers: List[int],
    is_conv_to_cpu: bool,
    is_check: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, is_conv_to_cpu, is_check, **kwargs)
    return model


def resnet50(is_conv_to_cpu=False, is_check=True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], is_conv_to_cpu, is_check,
                   **kwargs)