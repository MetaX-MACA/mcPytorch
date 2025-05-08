import torch
import copy

class TestModule(torch.nn.Module):
    def __init__(self, in_dims, out_dims, pool_dims, pool_kernel, stride):
        super().__init__()
        self.qkv = torch.nn.Linear(in_dims, in_dims, bias=True)
        pool_padding = [k // 2 for k in pool_kernel]
        self.conv2d = torch.nn.Conv2d(pool_dims, pool_dims, pool_kernel, stride=stride, padding=pool_padding, groups=pool_dims, bias=False)

    def forward(self, x):
        qkv = self.qkv(x).reshape(512, 96, 28, 28)
        q = self.conv2d(qkv)
        return q

def check_close(infer_result_data, golden_data, eps=1e-4):
    diff = infer_result_data - golden_data
    diff_square = diff * diff
    infer_result_square_double = 2 * infer_result_data * infer_result_data
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    print(f"check close error: {result}")
    return result < eps

def test_conv_depthwise2d_backward_3_2(dtype):
    # define input
    input = torch.rand(512, 196, 384, dtype=dtype).cuda().to(memory_format=torch.contiguous_format)
    input_cpu = copy.deepcopy(input).cpu().to(dtype)
    input_cpu.requires_grad = True
    input_cuda = copy.deepcopy(input).cuda().to(dtype)
    input_cuda.requires_grad = True

    m1 = TestModule(384, 768, pool_dims=96, pool_kernel=(3,3), stride=2)
    m1_cpu = copy.deepcopy(m1).cpu().to(dtype)
    m1_cuda = copy.deepcopy(m1).cuda().to(dtype)
    output_cpu = m1_cpu(input_cpu)
    output_cpu.backward(torch.ones(output_cpu.shape))
    output_cuda = m1_cuda(input_cuda)
    output_cuda.backward(torch.ones(output_cuda.shape).cuda())
    assert (check_close(output_cpu, output_cuda.float().cpu(), 1e-6))
    assert (check_close(input_cpu.grad, input_cuda.grad.float().cpu(), 1e-6))


def test_conv_depthwise2d_backward_3_4(dtype):
    # define input
    input = torch.rand(512, 196, 384, dtype=dtype).cuda().to(memory_format=torch.contiguous_format)
    input_cpu = copy.deepcopy(input).cpu().to(dtype)
    input_cpu.requires_grad = True
    input_cuda = copy.deepcopy(input).cuda().to(dtype)
    input_cuda.requires_grad = True

    m2 = TestModule(384, 768, pool_dims=96, pool_kernel=(3,3), stride=4)
    m2_cpu = copy.deepcopy(m2).cpu().to(dtype)
    m2_cuda = copy.deepcopy(m2).cuda().to(dtype)
    output_cpu = m2_cpu(input_cpu)
    output_cpu.backward(torch.ones(output_cpu.shape))
    output_cuda = m2_cuda(input_cuda)
    output_cuda.backward(torch.ones(output_cuda.shape).cuda())
    assert (check_close(output_cpu, output_cuda.float().cpu(), 1e-6))
    assert (check_close(input_cpu.grad, input_cuda.grad.float().cpu(), 1e-6))

test_conv_depthwise2d_backward_3_2(torch.float32)
test_conv_depthwise2d_backward_3_4(torch.float32)
