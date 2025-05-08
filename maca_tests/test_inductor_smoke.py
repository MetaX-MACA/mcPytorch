import torch


rtol = 1e-4
atol = 1e-4


def test_basic():
    def func(x):
        return torch.sin(x) + torch.cos(x)

    @torch.compile
    def func1(x):
        return func(x)

    # 1. functional call
    func_opt = torch.compile(func, backend="inductor")

    # 2. decorator
    func_opt1 = func1

    input = torch.randn([256, 256])

    g = func(input)
    out = func_opt(input.cuda())
    out1 = func_opt1(input.cuda())

    assert torch.allclose(out.cpu(), g, rtol, atol)
    assert torch.allclose(out1.cpu(), g, rtol, atol)


def test_max_autotune():
    def func(x):
        return torch.matmul(x, x) + x

    # 3. autotune
    func_opt = torch.compile(func, mode="max-autotune")

    input = torch.randn([256, 256])
    g = func(input)
    out = func_opt(input.cuda())

    assert torch.allclose(out.cpu(), g, rtol, atol)


def test_device_assert():
    def func(input, index):
        # torch.gather would call device_assert for bound check
        return torch.gather(input, 1, index)

    # 4. device_assert
    func_opt = torch.compile(func)

    device = "cuda:0"
    input = torch.rand((4, 128))
    index = torch.tensor([[0, 1], [3, 4]])
    output = func_opt(input.to(device), index.to(device))
    output_golden = func(input, index)
    assert torch.allclose(output.cpu(), output_golden, rtol, atol)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
