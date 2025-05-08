import os
import shutil
import torch
import torch.nn as nn
from torch import Tensor
import configs.config_resnet50 as config_resnet50
import configs.config_resnet50_all as config_resnet50_all
import configs.config_resnet50_check_test_data as config_resnet50_check_test_data
import configs.config_resnet50_create_golden_data as config_resnet50_create_golden_data
import configs.config_resnet50_c500_chip as config_resnet50_c500_chip
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import time

mode = "ci_mode"
config = config_resnet50
type_hardware = False

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda")
golden_or_check_data = 'check_data'
batch_size = 1
resnet50_model_path = config.resnet50_model_path
resnet50_model_name = config.resnet50_model_name
write_input_path = config.write_input_path
write_golden_input_tensor = config.write_input_path + 'test_batch' + str(batch_size) + '/'
write_golden_path = config.write_golden_path
write_golden_data_fwd_path = write_golden_path + 'test_batch' + str(batch_size) + '/golden_data_fwd/'
write_golden_data_bwd_path = write_golden_path + 'test_batch' + str(batch_size) + '/golden_data_bwd/'
read_input_path = config.read_input_path
read_golden_input_tensor = config.read_input_path + 'test_batch' + str(batch_size) + '/'
read_golden_path = config.read_golden_path
read_golden_data_fwd_path = read_golden_path + 'test_batch' + str(batch_size) + '/golden_data_fwd/'
read_golden_data_bwd_path = read_golden_path + 'test_batch' + str(batch_size) + '/golden_data_bwd/'
result_path = config.result_path
result_data_fwd_path = result_path + 'test_batch' + str(batch_size) + '/result_data_fwd/'
result_data_bwd_path = result_path + 'test_batch' + str(batch_size) + '/result_data_bwd/'

error_eps = 1e-3
index_fwd = 0
index_bwd = 0


def checkout_error(golden_data, check_data, file_name, device_name):
    if golden_data is None and check_data is None:
        print('Check ' + device_name + ' correct: ' + file_name)
    elif golden_data is None and check_data is not None:
        print('Error op ' + device_name + ' of ' + file_name + ' is out of eps!')
        print('Error Name op ' + device_name + ' :', file_name)
        quit()
    elif golden_data is not None and check_data is None:
        print('Error op ' + device_name + ' of ' + file_name + ' is out of eps!')
        print('Error Name op ' + device_name + ' :', file_name)
        quit()
    else:
        print('shape:', check_data.shape, ' of ' + device_name + ' of ' + file_name)
        if ((abs(golden_data.to(device_cpu) - check_data.to(device_cpu)).sum()) / (golden_data.numel()) > error_eps):
            print('Error op ' + device_name + ' of ' + file_name + ' is out of eps!')
            print('Error Name op ' + device_name + ' :', file_name)
            print('Error :', (abs(golden_data.to(device_cpu) - check_data.to(device_cpu)).sum()) / (golden_data.numel()))
            quit()
        else:
            print('Check ' + device_name + ' correct: ' + file_name + ', eps:',
                  (abs(golden_data.to(device_cpu) - check_data.to(device_cpu)).sum()) / (golden_data.numel()))


def hook_forward_fn(module, input, output):
    file_device = write_golden_data_fwd_path
    os.makedirs(file_device, exist_ok=True)

    global index_fwd
    index_file = index_fwd
    index_fwd = index_fwd + 1
    file_name_input = str(index_file) + '_' + str(module) + '_input' + '.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output' + '.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    if config.is_check_from_hook is True or (config.is_check_output is True and index_file is 157):
        torch.save(input[0], file_path_input)
        torch.save(output[0], file_path_output)


def hook_forward_fn_conv(module, input, output):
    if golden_or_check_data == 'golden_data':
        file_device = write_golden_data_fwd_path
        os.makedirs(file_device, exist_ok=True)
    if golden_or_check_data == 'check_data':
        file_device = result_data_fwd_path
        os.makedirs(file_device, exist_ok=True)

    global index_fwd
    index_file = index_fwd
    index_fwd = index_fwd + 1
    file_name = str(index_file) + '_' + str(module)
    file_name_input = str(index_file) + '_' + str(module) + '_input.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    if config.is_check_from_hook is True or (config.is_check_output is True and index_file is 157):
        input_write = input[0].to(device_cpu)
        output_write = output[0].to(device_cpu)
        torch.save(input_write, file_path_input)
        torch.save(output_write, file_path_output)
        conv_input = torch.load(read_golden_data_fwd_path + file_name_input, map_location="cpu")
        conv_output = torch.load(read_golden_data_fwd_path + file_name_output, map_location="cpu")
        checkout_error(conv_input, input_write, file_name, 'input fwd')
        checkout_error(conv_output, output_write, file_name, 'output fwd')

    if golden_or_check_data == 'check_data':
        log_path = result_path + 'test_batch' + str(batch_size) + '/log_msg.log'
        if os.path.exists(log_path) is False:
            os.mknod(log_path)
        with open(log_path, 'a') as logflie:
            logflie.write(file_name + '\n')


def hook_backward_fn(module, input, output):
    file_device = write_golden_data_bwd_path
    os.makedirs(file_device, exist_ok=True)

    global index_bwd
    index_file = index_bwd
    index_bwd = index_bwd + 1
    file_name_input = str(index_file) + '_' + str(module) + '_input' + '.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output' + '.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    if config.is_check_from_hook is True or (config.is_check_output is True and index_file is 157):
        torch.save(input, file_path_input)
        torch.save(output, file_path_output)


def hook_backward_fn_conv(module, input, output):
    if golden_or_check_data == 'golden_data':
        file_device = write_golden_data_bwd_path
        os.makedirs(file_device, exist_ok=True)
    if golden_or_check_data == 'check_data':
        file_device = result_data_bwd_path
        os.makedirs(file_device, exist_ok=True)

    global index_bwd
    index_file = index_bwd
    index_bwd = index_bwd + 1
    file_name = str(index_file) + '_' + str(module)
    file_name_input = str(index_file) + '_' + str(module) + '_input.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    if config.is_check_from_hook is True or (config.is_check_output is True and index_file is 157):
        input_write = input
        output_write = output
        torch.save(input_write, file_path_input)
        torch.save(output_write, file_path_output)
        conv_input = torch.load(read_golden_data_bwd_path + file_name_input, map_location="cpu")
        conv_output = torch.load(read_golden_data_bwd_path + file_name_output, map_location="cpu")
        for idx in range(len(input)):
            if "AdaptiveAvgPool2d" in str(module):
                break
            checkout_error(conv_input[idx], input_write[idx], file_name, 'input bwd' + str(idx))
            if str(module)[0:11] == 'BatchNorm2d':
                break
        for idx in range(len(output)):
            checkout_error(conv_output[idx], output_write[idx], file_name, 'output bwd' + str(idx))
            if str(module)[0:11] == 'BatchNorm2d':
                break

    if golden_or_check_data == 'check_data':
        log_path = result_path + 'test_batch' + str(batch_size) + '/log_msg.log'
        if os.path.exists(log_path) is False:
            os.mknod(log_path)
        with open(log_path, 'a') as logflie:
            logflie.write(file_name + '\n')


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
        if type_hardware == False:
            self.register_hook()

    def register_hook(self):
        list_op = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear']
        for name_op, type_op in vars(self)['_modules'].items():
            if type_op.__class__.__name__ in list_op:
                if self.is_check:
                    vars(self)['_modules'][name_op].register_forward_hook(hook_forward_fn_conv)
                    vars(self)['_modules'][name_op].register_backward_hook(hook_backward_fn_conv)
                else:
                    vars(self)['_modules'][name_op].register_forward_hook(hook_forward_fn)
                    vars(self)['_modules'][name_op].register_backward_hook(hook_backward_fn)
            elif type_op.__class__.__name__ == 'Sequential':
                for bottlenect_module in vars(self)['_modules'][name_op]:
                    for bottlenect_name_op, bottlenect_type_op in vars(bottlenect_module)['_modules'].items():
                        if bottlenect_type_op.__class__.__name__ in list_op:
                            if self.is_check:
                                vars(bottlenect_module)['_modules'][bottlenect_name_op].register_forward_hook(
                                    hook_forward_fn_conv)
                                vars(bottlenect_module)['_modules'][bottlenect_name_op].register_backward_hook(
                                    hook_backward_fn_conv)
                            else:
                                vars(bottlenect_module)[
                                    '_modules'][bottlenect_name_op].register_forward_hook(hook_forward_fn)
                                vars(bottlenect_module)[
                                    '_modules'][bottlenect_name_op].register_backward_hook(hook_backward_fn)
                        elif bottlenect_type_op.__class__.__name__ == 'Sequential':
                            for downsample_Seq in vars(bottlenect_module)['_modules'][bottlenect_name_op]:
                                if self.is_check:
                                    downsample_Seq.register_forward_hook(hook_forward_fn_conv)
                                    downsample_Seq.register_backward_hook(hook_backward_fn_conv)
                                else:
                                    downsample_Seq.register_forward_hook(hook_forward_fn)
                                    downsample_Seq.register_backward_hook(hook_backward_fn)

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


def download_resnet50_model():
    if os.path.exists(config.resnet50_model_download_path + "resnet50-0676ba61.pth") is False:
        load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth',
                                model_dir=config.resnet50_model_download_path)


def update_batch_size(input_batch, input_mode, input_type_hardware):
    global mode
    mode = input_mode
    global config
    if mode == "create_golden_data":
        config = config_resnet50_create_golden_data
    elif mode == "check_test_data":
        config = config_resnet50_check_test_data
    elif mode == "ci_mode":
        config = config_resnet50
    elif mode == "all":
        config = config_resnet50_all
    elif mode == "c500_chip":
        config = config_resnet50_c500_chip
    global type_hardware
    type_hardware = input_type_hardware
    global batch_size
    batch_size = input_batch
    global write_input_path
    write_input_path = config.write_input_path
    global write_golden_input_tensor
    write_golden_input_tensor = write_input_path + 'test_batch' + str(batch_size) + '/'
    global write_golden_path
    write_golden_path = config.write_golden_path
    global write_golden_data_fwd_path
    write_golden_data_fwd_path = write_golden_path + 'test_batch' + str(batch_size) + '/golden_data_fwd/'
    global write_golden_data_bwd_path
    write_golden_data_bwd_path = write_golden_path + 'test_batch' + str(batch_size) + '/golden_data_bwd/'
    global read_input_path
    read_input_path = config.read_input_path
    global read_golden_input_tensor
    read_golden_input_tensor = read_input_path + 'test_batch' + str(batch_size) + '/'
    global read_golden_path
    read_golden_path = config.read_golden_path
    global read_golden_data_fwd_path
    read_golden_data_fwd_path = read_golden_path + 'test_batch' + str(batch_size) + '/golden_data_fwd/'
    global read_golden_data_bwd_path
    read_golden_data_bwd_path = read_golden_path + 'test_batch' + str(batch_size) + '/golden_data_bwd/'
    global result_data_fwd_path
    result_data_fwd_path = result_path + 'test_batch' + str(batch_size) + '/result_data_fwd/'
    global result_data_bwd_path
    result_data_bwd_path = result_path + 'test_batch' + str(batch_size) + '/result_data_bwd/'
    global resnet50_model_path
    resnet50_model_path = config.resnet50_model_path
    global resnet50_model_name
    resnet50_model_name = config.resnet50_model_name


def check_golden_file():
    if os.path.exists(resnet50_model_path + resnet50_model_name) is False:
        ret = False
        print("Error: Missing resnet50 model file! Please download model file.")
    if os.path.exists(read_golden_input_tensor + 'input_tensor.pth') is False:
        ret = False
        print("Error: Missing input tensor file! Please generate input tensor.")
    if os.path.exists(read_golden_input_tensor + 'label_tensor.pth') is False:
        ret = False
        print("Error: Missing label tensor file! Please generate input tensor.")
    if os.path.exists(read_golden_data_fwd_path) is False:
        ret = False
        print("Error: Missing golden data fwd file! Please generate golden data fwd file.")
    if (config.is_backward):
        if os.path.exists(read_golden_data_bwd_path) is False:
            ret = False
            print("Error: Missing golden data bwd file! Please generate golden data bwd file.")


def check_create_golden_file():
    if os.path.exists(resnet50_model_path + resnet50_model_name) is False:
        ret = False
        print("Error: Missing resnet50 model file! Please download model file.")
    if os.path.exists(read_golden_input_tensor + 'input_tensor.pth') is False:
        ret = False
        print("Error: Missing input tensor file! Please generate input tensor.")
    if os.path.exists(read_golden_input_tensor + 'label_tensor.pth') is False:
        ret = False
        print("Error: Missing label tensor file! Please generate input tensor.")
    if os.path.exists(write_golden_data_fwd_path):
        print("WARNING: This gloden data fwd file is exists! Result will be covered")
        shutil.rmtree(write_golden_data_fwd_path)
    else:
        print("WARNING: This golden data fwd file will create!")
    if (config.is_backward):
        if os.path.exists(write_golden_data_bwd_path):
            print("WARNING: This gloden data bwd file is exists! Result will be covered")
            shutil.rmtree(write_golden_data_bwd_path)
        else:
            print("WARNING: This golden data bwd file will create!")


def check_test_file():
    if os.path.exists(result_data_fwd_path):
        print("WARNING: This result data fwd file is exists! Result will be covered")
        shutil.rmtree(result_data_fwd_path)
    if (config.is_backward):
        if os.path.exists(result_data_bwd_path):
            print("WARNING: This result data bwd file is exists! Result will be covered")
            shutil.rmtree(result_data_bwd_path)
    log_path = result_path + 'test_batch' + str(batch_size) + '/log_msg.log'
    if os.path.exists(log_path):
        print("WARNING: This log message is exists! Log message will be covered")
        os.remove(log_path)


def create_input_tensor(input_batch):
    if os.path.exists(write_golden_input_tensor + "input_tensor.pth"):
        print("WARNING: Input tensor will be covered")
        os.remove(write_golden_input_tensor + "input_tensor.pth")
    if os.path.exists(write_golden_input_tensor) is False:
        os.makedirs(write_golden_input_tensor)
    input_tensor = torch.rand(input_batch, 3, 224, 224, dtype=torch.float32)
    torch.save(input_tensor, write_golden_input_tensor + "input_tensor.pth")


def create_label_tensor(input_batch):
    if os.path.exists(write_golden_input_tensor + "label_tensor.pth"):
        print("WARNING: Label tensor will be covered")
        os.remove(write_golden_input_tensor + "label_tensor.pth")
    if os.path.exists(write_golden_input_tensor) is False:
        os.makedirs(write_golden_input_tensor)
    label_tensor = torch.empty(input_batch, dtype=torch.long).random_(1000)
    torch.save(label_tensor, write_golden_input_tensor + "label_tensor.pth")


def load_input_tensor():
    if os.path.exists(read_golden_input_tensor + "input_tensor.pth") is False:
        print("Error: Missing input tensor file! Please generate input tensor.")
    input_resnet = torch.load(read_golden_input_tensor + 'input_tensor.pth', map_location="cpu")
    return input_resnet


def load_label_tensor():
    if os.path.exists(read_golden_input_tensor + "label_tensor.pth") is False:
        print("Error: Missing label tensor file! Please generate input tensor.")
    label_resnet = torch.load(read_golden_input_tensor + 'label_tensor.pth', map_location="cpu")
    return label_resnet


def resnet50_cpu(input_resnet_cpu, label_resnet_cpu):
    global index_fwd
    global index_bwd
    index_fwd = 0
    index_bwd = 0
    input_resnet_cpu = input_resnet_cpu.to(torch.float64)
    input_resnet_cpu.requires_grad_()
    resnet50_module_cpu = resnet50(is_conv_to_cpu=False, is_check=False)
    resnet50_module_cpu.load_state_dict(torch.load(resnet50_model_path + resnet50_model_name))
    resnet50_module_cpu.to(torch.float64)
    output_resnet_cpu = resnet50_module_cpu(input_resnet_cpu)
    if (config.is_backward):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_resnet_cpu, label_resnet_cpu)
        loss.backward()
    return output_resnet_cpu


def resnet50_gpu(input_resnet_gpu, label_resnet_gpu):
    global index_fwd
    global index_bwd
    index_fwd = 0
    index_bwd = 0
    input_resnet_gpu = input_resnet_gpu.to(torch.float64)
    input_resnet_gpu = input_resnet_gpu.to(device_gpu)
    input_resnet_gpu.requires_grad_()
    label_resnet_gpu = label_resnet_gpu.to(device_gpu)
    resnet50_module_gpu = resnet50(is_conv_to_cpu=False, is_check=False).to(device_gpu)
    resnet50_module_gpu.load_state_dict(torch.load(resnet50_model_path + resnet50_model_name))
    resnet50_module_gpu.to(torch.float64)
    output_resnet_gpu = resnet50_module_gpu(input_resnet_gpu)
    if (config.is_backward):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_resnet_gpu, label_resnet_gpu).to(device_gpu)
        loss.backward()
    return output_resnet_gpu

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("trace_resnet.json")

def run_one_iter(resnet50_module_data, input_resnet_data, label_resnet_data, config):
    output_resnet_data = resnet50_module_data(input_resnet_data)
    if (config.is_backward):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_resnet_data, label_resnet_data.to(device_gpu))
        loss.backward()
    return output_resnet_data

def resnet50_data(input_resnet_data, label_resnet_data, conv_to_cpu, flag, flag_eps=1e-3):
    global golden_or_check_data
    golden_or_check_data = flag
    global error_eps
    error_eps = flag_eps
    global index_fwd
    global index_bwd
    index_fwd = 0
    index_bwd = 0
    input_resnet_data = input_resnet_data.to(device_gpu)
    input_resnet_data.requires_grad_()
    resnet50_module_data = resnet50(is_conv_to_cpu=conv_to_cpu, is_check=True).to(device_gpu)
    if type_hardware == False:
        resnet50_module_data.load_state_dict(torch.load(resnet50_model_path + resnet50_model_name))

        print("matmul.allow_tf32: ", torch.backends.cuda.matmul.allow_tf32)
        print("cudnn.allow_tf32: ", torch.backends.cudnn.allow_tf32)

    enable_dump_cuda_profile_data = True if "MACA_SAMPLE_RESNET50_ENABLE_PROFILE_DUMP" in os.environ else False

    if enable_dump_cuda_profile_data is True:
        warmup_cnt = 10
        for i in range(warmup_cnt):
            ts = time.time()
            output_resnet_data = run_one_iter(resnet50_module_data, input_resnet_data, label_resnet_data, config)
            te = time.time()
            print("iter time: ", te - ts)

        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=trace_handler) as prof:
            output_resnet_data = run_one_iter(resnet50_module_data, input_resnet_data, label_resnet_data, config)
    else:
        output_resnet_data = run_one_iter(resnet50_module_data, input_resnet_data, label_resnet_data, config)


    return output_resnet_data
