import os
import torch
import shutil
import numpy as np
import torch.nn as nn
import dlrm_data_generater
import configs.config_dlrm as config_dlrm

file_path = os.path.abspath(os.path.dirname((__file__)))
batch_size = 1
result_file_path = file_path + "/result/batch_size" + str(batch_size)
golden_fwd_file_path = result_file_path + "/golden_data_fwd/"
golden_bwd_file_path = result_file_path + "/golden_data_bwd/"
test_fwd_file_path = result_file_path + "/test_data_fwd/"
test_bwd_file_path = result_file_path + "/test_data_bwd/"
test_mode = ""
index_fwd = 0
index_bwd = 0
type_hardware = False


def checkout_error(golden_data, test_data, file_name):
    if isinstance(golden_data, torch.Tensor):
        if golden_data.is_sparse:
            real_eps = (abs(golden_data.cpu().to_dense() - test_data.cpu().to_dense()).sum()) / \
                (golden_data.cpu().to_dense().numel())
        else:
            real_eps = (abs(golden_data.cpu() - test_data.cpu()).sum()) / (golden_data.cpu().numel())
        if (real_eps > config_dlrm.eps):
            print("Error:", file_name + "_" + str(0))
            print("Error eps:", real_eps)
            quit()
        else:
            print("Correct:", file_name + "_" + str(0), ", Eps:", real_eps)
    else:
        for idx in range(len(golden_data)):
            if golden_data[idx] is None and test_data[idx] is None:
                print("Correct:", file_name + "_" + str(idx), ", Eps: 0")
            elif golden_data[idx] is None and test_data[idx] is not None:
                print("Error:", file_name + "_" + str(idx))
                print("Golden data is None, but test data is not None.")
                quit()
            elif golden_data[idx] is not None and test_data[idx] is None:
                print("Error:", file_name + "_" + str(idx))
                print("Golden data is not None, but test data is None.")
                quit()
            else:
                if golden_data[idx].is_sparse:
                    real_eps = (abs(golden_data[idx].cpu().to_dense() - test_data[idx].cpu().to_dense()).sum()) / \
                        (golden_data[idx].cpu().to_dense().numel())
                else:
                    real_eps = (abs(golden_data[idx].cpu() - test_data[idx].cpu()).sum()) / \
                        (golden_data[idx].cpu().numel())
                if (real_eps > config_dlrm.eps):
                    print("Error:", file_name + "_" + str(idx))
                    print("Error eps:", real_eps)
                    quit()
                else:
                    print("Correct:", file_name + "_" + str(idx), ", Eps:", real_eps)


def hook_forward_fn(module, input, output):
    global index_fwd
    index_file = index_fwd
    index_fwd = index_fwd + 1
    file_name_input = str(index_file) + "_" + str(module) + "_input.pth"
    file_name_output = str(index_file) + "_" + str(module) + "_output.pth"
    if test_mode == "golden_mode":
        golden_file_path_input = golden_fwd_file_path + file_name_input
        golden_file_path_output = golden_fwd_file_path + file_name_output
        torch.save(input, golden_file_path_input)
        torch.save(output, golden_file_path_output)
    elif test_mode == "test_mode":
        golden_file_path_input = golden_fwd_file_path + file_name_input
        golden_file_path_output = golden_fwd_file_path + file_name_output
        golden_input = torch.load(golden_file_path_input)
        golden_output = torch.load(golden_file_path_output)
        checkout_error(golden_input, input, file_name_input[0: -4])
        checkout_error(golden_output, output, file_name_output[0: -4])
        test_file_path_input = test_fwd_file_path + file_name_input
        test_file_path_output = test_fwd_file_path + file_name_output
        torch.save(input, test_file_path_input)
        torch.save(output, test_file_path_output)
    else:
        print("ERROR: Test mode of" + test_mode + "is not supported")
        quit()


def hook_backward_fn(module, input, output):
    global index_bwd
    index_file = index_bwd
    index_bwd = index_bwd + 1
    file_name_input = str(index_file) + "_" + str(module) + "_input.pth"
    file_name_output = str(index_file) + "_" + str(module) + "_output.pth"
    if test_mode == "golden_mode":
        golden_file_path_input = golden_bwd_file_path + file_name_input
        golden_file_path_output = golden_bwd_file_path + file_name_output
        torch.save(input, golden_file_path_input)
        torch.save(output, golden_file_path_output)
    elif test_mode == "test_mode":
        golden_file_path_input = golden_bwd_file_path + file_name_input
        golden_file_path_output = golden_bwd_file_path + file_name_output
        golden_input = torch.load(golden_file_path_input)
        golden_output = torch.load(golden_file_path_output)
        checkout_error(golden_input, input, file_name_input[0: -4])
        checkout_error(golden_output, output, file_name_output[0: -4])
        test_file_path_input = test_bwd_file_path + file_name_input
        test_file_path_output = test_bwd_file_path + file_name_output
        torch.save(input, test_file_path_input)
        torch.save(output, test_file_path_output)
    else:
        print("ERROR: Test mode of" + test_mode + "is not supported")
        quit()


class DLRM_Net(nn.Module):
    def __init__(self):
        super(DLRM_Net, self).__init__()
        np.random.seed(config_dlrm.rand_seed)
        self.embedding_num = config_dlrm.embedding_num
        self.embedding_dim = config_dlrm.embedding_dim
        self.embedding_mode = config_dlrm.embedding_mode
        self.interaction_mode = config_dlrm.interaction_mode
        self.layer_bot = config_dlrm.layer_bot + [self.embedding_dim]
        if self.interaction_mode == "dot":
            self.layer_top = [int(len(self.embedding_num) * (len(self.embedding_num) + 1) /
                                  2) + self.embedding_dim] + config_dlrm.layer_top
        else:
            self.layer_top = [self.embedding_dim * (len(self.embedding_num) + 1)] + config_dlrm.layer_top
        self.emb_l = self.create_emb(self.embedding_dim, self.embedding_num)
        self.bot_l = self.create_mlp(self.layer_bot, -1)
        self.top_l = self.create_mlp(self.layer_top, len(self.layer_top) - 2)
        self.loss_threshold = config_dlrm.loss_threshold
        if config_dlrm.loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif config_dlrm.loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        else:
            print("ERROR: Loss_function of" + config_dlrm.loss_function + "is not supported")
            quit()
        if type_hardware == False:
            self.register_hook()

    def register_hook(self):
        for name_op, type_op in vars(self)["_modules"].items():
            if type_op.__class__.__name__ == "Sequential" or type_op.__class__.__name__ == "ModuleList":
                for sub_module in vars(self)["_modules"][name_op]:
                    sub_module.register_forward_hook(hook_forward_fn)
                    sub_module.register_backward_hook(hook_backward_fn)
            else:
                type_op.register_forward_hook(hook_forward_fn)
                type_op.register_backward_hook(hook_backward_fn)

    def create_mlp(self, ln, sigmoid_layer):
        layers = nn.ModuleList()
        for i in range(0, len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, len(ln)):
            n = ln[i]
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)
            emb_l.append(EE)
        return emb_l

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=None,
            )
            ly.append(V)
        return ly

    def interact_features(self, x, ly):
        if self.interaction_mode == "dot":
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.interaction_mode == "cat":
            R = torch.cat([x] + ly, dim=1)
        else:
            print("ERROR: Interaction_mode of" + self.interaction_mode + "is not supported")
            quit()
        return R

    def forward(self, dense_x, lS_o, lS_i):
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        x = self.apply_mlp(dense_x, self.bot_l)
        z = self.interact_features(x, ly)
        p = self.apply_mlp(z, self.top_l)
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p
        return z


def create_input_data(batch_size):
    input_data_iter = dlrm_data_generater.make_random_data_and_loader(
        batch_size, config_dlrm.embedding_num, config_dlrm.layer_bot[0])
    input_data = next(iter(input_data_iter))
    return input_data[0], input_data[1], input_data[2], input_data[3]


def create_result_file():
    if os.path.exists(result_file_path):
        print("WARNING: This result data is exists! Result data will be recover")
        shutil.rmtree(result_file_path)
    os.makedirs(golden_fwd_file_path)
    os.makedirs(golden_bwd_file_path)
    os.makedirs(test_fwd_file_path)
    os.makedirs(test_bwd_file_path)


def set_batch_size(input_batch_size):
    global batch_size
    batch_size = input_batch_size
    global result_file_path
    result_file_path = file_path + "/result/batch_size" + str(batch_size)
    global golden_fwd_file_path
    golden_fwd_file_path = result_file_path + "/golden_data_fwd/"
    global golden_bwd_file_path
    golden_bwd_file_path = result_file_path + "/golden_data_bwd/"
    global test_fwd_file_path
    test_fwd_file_path = result_file_path + "/test_data_fwd/"
    global test_bwd_file_path
    test_bwd_file_path = result_file_path + "/test_data_bwd/"


def set_test_mode(input_mode, input_type_hardware):
    global test_mode
    test_mode = input_mode
    global index_fwd
    index_fwd = 0
    global index_bwd
    index_bwd = 0
    global type_hardware
    type_hardware = input_type_hardware
