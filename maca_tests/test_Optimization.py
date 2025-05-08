import torch
import torch.nn as nn


emb_type = 10
emb_size = 3
input_size = 15
eps = 1e-5


class EmbeddingNet(nn.Module):
    def __init__(self, sparse_flag: bool = False):
        super(EmbeddingNet, self).__init__()
        self.emb_type = emb_type
        self.emb_size = emb_size
        self.input_size = input_size
        self.embedding = nn.Embedding(self.emb_type, self.emb_size, sparse=sparse_flag)

    def forward(self, x):
        out = self.embedding(x)
        return out


def test_Adadelta():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.Adadelta(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.Adadelta(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test Adadelta pass, error eps:", error)
        else:
            ret = False
            print("Error test Adadelta failed, error eps:", error)
            break
    return ret


def test_Adagrad():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.Adagrad(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.Adagrad(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test Adagrad pass, error eps:", error)
        else:
            ret = False
            print("Error test Adagrad failed, error eps:", error)
            break
    return ret


def test_Adam():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.Adam(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test Adam pass, error eps:", error)
        else:
            ret = False
            print("Error test Adam failed, error eps:", error)
            break
    return ret


def test_AdamW():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.AdamW(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test AdamW pass, error eps:", error)
        else:
            ret = False
            print("Error test AdamW failed, error eps:", error)
            break
    return ret


def test_SparseAdam():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet(sparse_flag=True)
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.SparseAdam(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet(sparse_flag=True).cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.SparseAdam(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test SparseAdam pass, error eps:", error)
        else:
            ret = False
            print("Error test SparseAdam failed, error eps:", error)
            break
    return ret


def test_Adamax():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.Adamax(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.Adamax(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test Adamax pass, error eps:", error)
        else:
            ret = False
            print("Error test Adamax failed, error eps:", error)
            break
    return ret


def test_ASGD():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.ASGD(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.ASGD(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test ASGD pass, error eps:", error)
        else:
            ret = False
            print("Error test ASGD failed, error eps:", error)
            break
    return ret


def test_LBFGS():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.LBFGS(model_cpu.parameters())

    def closure_cpu():
        output_tensor_cpu = model_cpu(input_tensor_cpu)
        loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
        loss_cpu.backward()
        optimizer_cpu.zero_grad()
        return loss_cpu
    optimizer_cpu.step(closure_cpu)

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.LBFGS(model_cpu.parameters())

    def closure_gpu():
        output_tensor_gpu = model_gpu(input_tensor_gpu)
        loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
        loss_gpu.backward()
        optimizer_gpu.zero_grad()
        return loss_gpu
    optimizer_gpu.step(closure_gpu)

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test LBFGS pass, error eps:", error)
        else:
            ret = False
            print("Error test LBFGS failed, error eps:", error)
            break
    return ret


def test_NAdam():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.NAdam(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.NAdam(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test NAdam pass, error eps:", error)
        else:
            ret = False
            print("Error test NAdam failed, error eps:", error)
            break
    return ret


def test_RAdam():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.RAdam(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.RAdam(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test RAdam pass, error eps:", error)
        else:
            ret = False
            print("Error test RAdam failed, error eps:", error)
            break
    return ret


def test_RMSprop():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.RMSprop(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.RMSprop(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test RMSprop pass, error eps:", error)
        else:
            ret = False
            print("Error test RMSprop failed, error eps:", error)
            break
    return ret


def test_Rprop():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.Rprop(model_cpu.parameters())
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.Rprop(model_cpu.parameters())
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test Rprop pass, error eps:", error)
        else:
            ret = False
            print("Error test Rprop failed, error eps:", error)
            break
    return ret


def test_SGD():
    input_tensor_cpu = (torch.rand(input_size) * 10).int()
    golden_cpu = torch.ones(input_size, emb_size)
    model_cpu = EmbeddingNet()
    model_cpu.train()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), 0.01)
    output_tensor_cpu = model_cpu(input_tensor_cpu)
    loss_cpu = criterion_cpu(output_tensor_cpu, golden_cpu)
    loss_cpu.backward()
    optimizer_cpu.zero_grad()
    optimizer_cpu.step()

    input_tensor_gpu = input_tensor_cpu.cuda()
    golden_gpu = golden_cpu.cuda()
    model_gpu = EmbeddingNet().cuda()
    model_gpu.train()
    criterion_gpu = nn.CrossEntropyLoss().cuda()
    optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), 0.01)
    output_tensor_gpu = model_gpu(input_tensor_gpu)
    loss_gpu = criterion_gpu(output_tensor_gpu, golden_gpu)
    loss_gpu.backward()
    optimizer_gpu.zero_grad()
    optimizer_gpu.step()

    ret = True
    for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(model_cpu.named_parameters(), model_cpu.named_parameters()):
        error = ((abs(param_cpu - param_gpu)).sum()) / param_cpu.numel()
        if error < eps:
            ret = True
            print("Check test SGD pass, error eps:", error)
        else:
            ret = False
            print("Error test SGD faileded, error eps:", error)
            break
    return ret


if __name__ == "__main__":
    ret = True
    ret = ret and test_Adadelta()
    ret = ret and test_Adagrad()
    ret = ret and test_Adam()
    ret = ret and test_AdamW()
    ret = ret and test_SparseAdam()
    ret = ret and test_Adamax()
    ret = ret and test_ASGD()
    ret = ret and test_LBFGS()
    ret = ret and test_NAdam()
    ret = ret and test_RAdam()
    ret = ret and test_RMSprop()
    ret = ret and test_Rprop()
    ret = ret and test_SGD()
    if ret is True:
        print("All optim test pass")
        exit(0)
    else:
        print("Optim test faileded")
        exit(1)
