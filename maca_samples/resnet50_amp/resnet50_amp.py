import torch
import torchvision
import argparse
import copy
import os
import shutil
import configs.config_resnet50_amp as config_resnet50_amp


# set parameter
ret = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, help="batch size")
parser.add_argument("--num_iteration", default=2, help="num iteration")
parser.add_argument("--num_epoch", default=10, help="num epoch")
parser.add_argument("--run_type", default="check_data", help="create_golden or check_data")
parser.add_argument("--debug", action="store_true", default=False, help="debug for check data")
parser.add_argument("--amp_mode", default="no_amp", help="amp or no_amp")
parser.add_argument("--checkpoint_mode", default="no_checkpoint", help="no_checkpoint or checkpoint_record or checkpoint_run")
parser.add_argument("--checkpoint_step", default=-1, help="checkpoint epoch")
parser.add_argument("--checkpoint_epoch", default=-1, help="checkpoint epoch")
parser.add_argument("--checkpoint_iteration", default=-1, help="checkpoint iteration")
batch_size = int(parser.parse_args().batch_size)
num_iteration = int(parser.parse_args().num_iteration)
num_epoch = int(parser.parse_args().num_epoch)
run_type = str(parser.parse_args().run_type)
debug = bool(parser.parse_args().debug)
amp_mode = str(parser.parse_args().amp_mode)
checkpoint_mode = str(parser.parse_args().checkpoint_mode)
checkpoint_step = int(parser.parse_args().checkpoint_step)
checkpoint_epoch = int(parser.parse_args().checkpoint_epoch)
checkpoint_iteration = int(parser.parse_args().checkpoint_iteration)
file_path = os.path.abspath(os.path.dirname((__file__)))
index_epoch = 0
index_iteration = 0
error_eps = config_resnet50_amp.error_eps
loss_eps = config_resnet50_amp.loss_eps
index_fwd = 0
index_bwd = 0
golden_data_path_fwd = ""
golden_data_path_bwd = ""
check_data_path_fwd = ""
check_data_path_bwd = ""
if amp_mode == "amp":
    flag_resnet50_amp_batch_size = "/resnet50_amp/batch_size"
    flag_result_resnet50_amp_batch_size = "/result/resnet50_amp/batch_size"
else:
    flag_resnet50_amp_batch_size = "/resnet50_no_amp/batch_size"
    flag_result_resnet50_amp_batch_size = "/result/resnet50_no_amp/batch_size"
if checkpoint_mode == "checkpoint_record":
    assert checkpoint_step != -1, "Please set checkpoint_step when checkpoint_mode is checkpoint_record!"
elif checkpoint_mode == "checkpoint_run":
    assert checkpoint_epoch != -1, "Please set checkpoint_epoch when checkpoint_mode is checkpoint_run!"
    assert checkpoint_iteration != -1, "Please set checkpoint_iteration when checkpoint_mode is checkpoint_run!"

def update_data_path():
    global golden_data_path_fwd
    golden_data_path_fwd = config_resnet50_amp.golden_data_path + flag_resnet50_amp_batch_size + str(batch_size) \
                        + '/debug/golden_data_fwd/epoch' + str(index_epoch) + "/iteration" + str(index_iteration) + "/"
    global golden_data_path_bwd
    golden_data_path_bwd = config_resnet50_amp.golden_data_path + flag_resnet50_amp_batch_size + str(batch_size) \
                        + '/debug/golden_data_bwd/epoch' + str(index_epoch) + "/iteration" + str(index_iteration) + "/"
    global check_data_path_fwd
    check_data_path_fwd = file_path + flag_result_resnet50_amp_batch_size + str(batch_size) + '/debug/result_data_fwd/epoch' \
                        + str(index_epoch) + "/iteration" + str(index_iteration) + "/"
    global check_data_path_bwd
    check_data_path_bwd = file_path + flag_result_resnet50_amp_batch_size + str(batch_size) + '/debug/result_data_bwd/epoch' \
                            + str(index_epoch) + "/iteration" + str(index_iteration) + "/"
update_data_path()


# check data function
def checkout_error(golden_data, check_data, file_name, device_name):
    print("epoch" + str(index_epoch) + "_iteration" + str(index_iteration) + ": " + file_name + " " + device_name)
    error_compute = (abs(golden_data.cpu() - check_data.cpu()).sum()) / (golden_data.numel())
    if debug == True:
        log_path = file_path + flag_result_resnet50_amp_batch_size + str(batch_size) + '/debug/log_msg_error.log'
    else:
        log_path = file_path + flag_result_resnet50_amp_batch_size + str(batch_size) + '/no_debug/log_msg_error.log'
    if os.path.exists(log_path) is False:
        os.mknod(log_path)
    with open(log_path, 'a') as logflie:
        log_info = "epoch" + str(index_epoch) + "_iteration" + str(index_iteration) + ":" \
                    + str(error_compute) + " (" + file_name + " " + device_name + ")"
        logflie.write(log_info + '\n')


# define hook functions
def hook_forward_create_golden(module, input, output):
    file_device = golden_data_path_fwd
    os.makedirs(file_device, exist_ok=True)
    global index_fwd
    index_file = index_fwd
    index_fwd = index_fwd + 1
    file_name_input = str(index_file) + '_' + str(module) + '_input' + '.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output' + '.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    torch.save(input[0], file_path_input)
    torch.save(output[0], file_path_output)

def hook_forward_check_data(module, input, output):
    file_device = check_data_path_fwd
    os.makedirs(file_device, exist_ok=True)
    global index_fwd
    index_file = index_fwd
    index_fwd = index_fwd + 1
    file_name = str(index_file) + '_' + str(module)
    file_name_input = str(index_file) + '_' + str(module) + '_input.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    input_write = input[0]
    output_write = output[0]
    torch.save(input_write, file_path_input)
    torch.save(output_write, file_path_output)
    golden_input = torch.load(golden_data_path_fwd + file_name_input, map_location="cuda")
    golden_output = torch.load(golden_data_path_fwd + file_name_output, map_location="cuda")
    checkout_error(golden_input, input_write, file_name, 'input fwd')
    checkout_error(golden_output, output_write, file_name, 'output fwd')

def hook_backward_create_golden(module, input, output):
    file_device = golden_data_path_bwd
    os.makedirs(file_device, exist_ok=True)
    global index_bwd
    index_file = index_bwd
    index_bwd = index_bwd + 1
    file_name_input = str(index_file) + '_' + str(module) + '_input' + '.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output' + '.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    torch.save(input, file_path_input)
    torch.save(output, file_path_output)

def hook_backward_check_data(module, input, output):
    file_device = check_data_path_bwd
    os.makedirs(file_device, exist_ok=True)
    global index_bwd
    index_file = index_bwd
    index_bwd = index_bwd + 1
    file_name = str(index_file) + '_' + str(module)
    file_name_input = str(index_file) + '_' + str(module) + '_input.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output.pth'
    file_path_input = file_device + file_name_input
    file_path_output = file_device + file_name_output
    input_write = input
    output_write = output
    torch.save(input_write, file_path_input)
    torch.save(output_write, file_path_output)
    golden_input = torch.load(golden_data_path_bwd + file_name_input, map_location="cuda")
    golden_output = torch.load(golden_data_path_bwd + file_name_output, map_location="cuda")
    for idx in range(len(input)):
        checkout_error(golden_input[idx], input_write[idx], file_name, 'input bwd' + str(idx))
    for idx in range(len(output)):
        checkout_error(golden_output[idx], output_write[idx], file_name, 'output bwd' + str(idx))

def register_hook(model, run_type):
    list_op = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear']
    for name_op, type_op in vars(model)['_modules'].items():
        if type_op.__class__.__name__ in list_op:
            if run_type == "check_data":
                vars(model)['_modules'][name_op].register_forward_hook(hook_forward_check_data)
                vars(model)['_modules'][name_op].register_backward_hook(hook_backward_check_data)
            else:
                vars(model)['_modules'][name_op].register_forward_hook(hook_forward_create_golden)
                vars(model)['_modules'][name_op].register_backward_hook(hook_backward_create_golden)
        elif type_op.__class__.__name__ == 'Sequential':
            for bottlenect_module in vars(model)['_modules'][name_op]:
                for bottlenect_name_op, bottlenect_type_op in vars(bottlenect_module)['_modules'].items():
                    if bottlenect_type_op.__class__.__name__ in list_op:
                        if run_type == "check_data":
                            vars(bottlenect_module)['_modules'][bottlenect_name_op].register_forward_hook(
                                hook_forward_check_data)
                            vars(bottlenect_module)['_modules'][bottlenect_name_op].register_backward_hook(
                                hook_backward_check_data)
                        else:
                            vars(bottlenect_module)[
                                '_modules'][bottlenect_name_op].register_forward_hook(hook_forward_create_golden)
                            vars(bottlenect_module)[
                                '_modules'][bottlenect_name_op].register_backward_hook(hook_backward_create_golden)
                    elif bottlenect_type_op.__class__.__name__ == 'Sequential':
                        for downsample_Seq in vars(bottlenect_module)['_modules'][bottlenect_name_op]:
                            if run_type == "check_data":
                                downsample_Seq.register_forward_hook(hook_forward_check_data)
                                downsample_Seq.register_backward_hook(hook_backward_check_data)
                            else:
                                downsample_Seq.register_forward_hook(hook_forward_create_golden)
                                downsample_Seq.register_backward_hook(hook_backward_create_golden)


# check file exit
if run_type == "create_golden":
    golden_path = config_resnet50_amp.golden_data_path + flag_resnet50_amp_batch_size + str(batch_size)
    if os.path.exists(golden_path):
        print("WARNING: This gloden data file is exists! Result will be covered")
        shutil.rmtree(golden_path)
else:  # check_data
    check_path = file_path + flag_result_resnet50_amp_batch_size + str(batch_size)
    if os.path.exists(check_path) and checkpoint_mode != "checkpoint_run":
        print("WARNING: This check data file is exists! Result will be covered")
        shutil.rmtree(check_path)


# init resnet50
class resnet50(torch.nn.Module):
    def __init__(self, net, run_type):
        super(resnet50, self).__init__()
        self.net = net
        if debug or run_type == "create_golden":
            register_hook(self.net, run_type)

    def forward(self, x):
        return self.net(x)

resnet50_torchvision = torchvision.models.resnet50(pretrained=False)
resnet50_torchvision.load_state_dict(torch.load(config_resnet50_amp.resnet50_model_path \
                                            + config_resnet50_amp.resnet50_model_name))
resnet50 = resnet50(resnet50_torchvision, run_type)
resnet50_cuda = copy.deepcopy(resnet50)
resnet50_cuda = resnet50_cuda.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer_cuda = torch.optim.SGD(resnet50_cuda.parameters(), 0.01)
resnet50_cuda.train()


# input tensor and label tensor
inp_label_dir_path = config_resnet50_amp.golden_data_path + \
                        flag_resnet50_amp_batch_size + str(batch_size)
inp_label_path = inp_label_dir_path + "/inp_label_data.pth"
if run_type == "create_golden":
    data_cuda = []
    for _ in range(num_iteration):
        inp_cpu = torch.rand(batch_size, 3, 224, 224).requires_grad_()
        label_cpu = torch.empty(batch_size, dtype=torch.long).random_(1000)
        inp_cuda = inp_cpu.clone().detach().cuda().requires_grad_()
        label_cuda = label_cpu.clone().detach().cuda()
        data_cuda.append([inp_cuda, label_cuda])
    os.makedirs(inp_label_dir_path, exist_ok=True)
    torch.save(data_cuda, inp_label_path)
else:  # check_data
    data_cuda = torch.load(inp_label_path, map_location="cuda")


# create golden data or check data of resnet50
def run_resnet50(inp_cuda, label_cuda, resnet50_cuda, criterion):
    out_cuda = resnet50_cuda(inp_cuda)
    loss_cuda = criterion(out_cuda, label_cuda)
    if run_type == "create_golden":
        log_loss_path = config_resnet50_amp.golden_data_path + flag_resnet50_amp_batch_size + \
                        str(batch_size) + '/log_msg_loss.log'
    else:
        log_loss_path = file_path + flag_result_resnet50_amp_batch_size + str(batch_size) + '/log_msg_loss.log'
    if os.path.exists(log_loss_path) is False:
        os.makedirs(file_path + flag_result_resnet50_amp_batch_size + str(batch_size), exist_ok=True)
        os.mknod(log_loss_path)
    with open(log_loss_path, 'a') as logflie:
        log_info = "loss: " + str(loss_cuda)
        logflie.write(log_info + '\n')
    if run_type == "create_golden" or debug == False:
        out_loss_golden_dir_path = config_resnet50_amp.golden_data_path + \
            flag_resnet50_amp_batch_size + str(batch_size) + "/no_debug/golden_data/epoch" + str(index_epoch) + \
            "/iteration" + str(index_iteration) + "/"
        out_loss_golden_path = out_loss_golden_dir_path + "/out_loss_data.pth"
        if run_type == "create_golden":
            os.makedirs(out_loss_golden_dir_path, exist_ok=True)
            torch.save([out_cuda, loss_cuda], out_loss_golden_path)
        else:  # check_data
            out_loss_data_dir_path = file_path + flag_result_resnet50_amp_batch_size + str(batch_size) \
                + "/no_debug/result_data/epoch" + str(index_epoch) + "/iteration" + \
                str(index_iteration) + "/"
            out_loss_data_path = out_loss_data_dir_path + "/out_loss_data.pth"
            os.makedirs(out_loss_data_dir_path, exist_ok=True)
            out_golden, loss_golden = torch.load(out_loss_golden_path, map_location="cuda")
            checkout_error(out_golden, out_cuda, "epoch" + str(index_epoch) + " iteration" + \
                            str(index_iteration), 'output tensor')
            checkout_error(loss_golden, loss_cuda, "epoch" + str(index_epoch) + " iteration" + \
                            str(index_iteration), 'loss tensor')
            torch.save([out_cuda, loss_cuda], out_loss_data_path)
    return loss_cuda

checkpoint_dir_path = file_path + flag_result_resnet50_amp_batch_size + str(batch_size) + '/checkpoint_cache/'
if checkpoint_mode == "checkpoint_record":
    epoch = 0
    i = 0
    step_i = 0
    if os.path.exists(checkpoint_dir_path) is False:
        os.makedirs(checkpoint_dir_path, exist_ok=True)
    else:
        os.rmdir(checkpoint_dir_path)
        os.makedirs(checkpoint_dir_path, exist_ok=True)
elif checkpoint_mode == "checkpoint_run":
    checkpoint_load_state_dir = checkpoint_dir_path + "epoch" + str(checkpoint_epoch) + "_iteration" + \
                    str(checkpoint_iteration) + ".pth"
    checkpoint_load_state = torch.load(checkpoint_load_state_dir)
    resnet50_cuda.load_state_dict(checkpoint_load_state["net"])
    optimizer_cuda.load_state_dict(checkpoint_load_state["optimizer"])
    epoch = checkpoint_load_state["epoch"] + 1
    i = (checkpoint_load_state["iteration"] + 1) % len(data_cuda)
    step_i = checkpoint_load_state["step_i"] + 1
    checkpoint_step = checkpoint_load_state["checkpoint_step"]
else:
    epoch = 0
    i = 0
    step_i = 0

while epoch < num_epoch:
    index_epoch = epoch
    while i < len(data_cuda):
        (inp_cuda, label_cuda) = data_cuda[i]
        index_iteration = i
        index_fwd = 0
        index_bwd = 0
        update_data_path()
        optimizer_cuda.zero_grad()
        # run resnet50 with amp
        if amp_mode == "amp":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss_cuda = run_resnet50(inp_cuda, label_cuda, resnet50_cuda, criterion)
        else:
            loss_cuda = run_resnet50(inp_cuda, label_cuda, resnet50_cuda, criterion)
        loss_cuda.backward()
        optimizer_cuda.step()
        if checkpoint_mode == "checkpoint_record" or checkpoint_mode == "checkpoint_run":
            if step_i % checkpoint_step == checkpoint_step - 1:
                checkpoint_state = {'net':resnet50_cuda.state_dict(), 'optimizer':optimizer_cuda.state_dict(), \
                                    'epoch':epoch, 'iteration':i, 'step_i':step_i, 'checkpoint_step':checkpoint_step}
                checkpoint_file_path = checkpoint_dir_path + "epoch" + str(epoch) + "_iteration" + str(i) + ".pth"
                if os.path.exists(checkpoint_dir_path) is False:
                    os.makedirs(checkpoint_dir_path, exist_ok=True)
                torch.save(checkpoint_state, checkpoint_file_path)
        i += 1
        step_i += 1
    i = 0
    epoch += 1


# result
if loss_cuda < loss_eps:
    ret = True
else:
    ret = False

if ret:
    print('###### resnet50_amp.py passed! ######')
    exit(0)
else:
    print('###### resnet50_amp.py failed! ######')
    exit(1)
