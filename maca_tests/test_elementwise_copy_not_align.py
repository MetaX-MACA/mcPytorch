import torch

torch.manual_seed(0)

def test_elementwise_copy_2_1(shape, dtype, i = 1):
    input = torch.randn(shape[0] * 2, shape[1] * 2, dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input[i:i+shape[0], i:i+shape[1]]
    input = input.as_strided(shape, (shape[1], 1))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, (1, shape[0]))

    input_c = input_c[i:i+shape[0], i:i+shape[1]]
    input_c = input_c.as_strided(shape, (shape[1], 1))
    output_c = output_c.as_strided(shape, (1, shape[0]))

    output_c.copy_(input_c)
    output.copy_(input)
    if not torch.allclose(output.cpu(), output_c):
        print("test_elementwise_copy_2_1 Error!")
        exit(1)

def test_elementwise_copy_3_1_trans_01(shape, dtype, i = 1):
    input = torch.randn(shape[0] * 2, shape[1] * 2, shape[2] * 2, dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input[i:i+shape[0], i:i+shape[1], i:i+shape[2]]
    input = input.as_strided(shape, (shape[1], 1, shape[0] * shape[1]))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, (1, shape[0], shape[0] * shape[1]))

    input_c = input_c[i:i+shape[0], i:i+shape[1], i:i+shape[2]]
    input_c = input_c.as_strided(shape, (shape[1], 1, shape[0] * shape[1]))
    output_c = output_c.as_strided(shape, (1, shape[0], shape[0] * shape[1]))

    output_c.copy_(input_c)
    output.copy_(input)
    if not torch.allclose(output.cpu(), output_c):
        print("test_elementwise_copy_3_1_trans_01 Error!")
        exit(1)

def test_elementwise_copy_3_1_trans_012(shape, dtype, i = 1):
    input = torch.randn(shape[0] * shape[1] * 2, shape[1] * 2, shape[2] * 2, dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input[i:i+shape[0], i:i+shape[1], i:i+shape[2]]
    # input = torch.randn(shape[0] * shape[1], shape[1], 2, dtype=dtype, device="cuda")
    input = input.as_strided(shape, (shape[0] * shape[1], shape[1], 1))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, (1, shape[0], shape[0] * shape[1]))

    input_c = input_c[i:i+shape[0], i:i+shape[1], i:i+shape[2]]
    input_c = input_c.as_strided(shape, (shape[0] * shape[1], shape[1], 1))
    output_c = output_c.as_strided(shape, (1, shape[0], shape[0] * shape[1]))

    output_c.copy_(input_c)
    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        print("test_elementwise_copy_3_1_trans_012 Error!")
        exit(1)

def test_elementwise_copy_3_1_trans_12(shape, dtype, i = 1):
    input = torch.randn(shape[0] * shape[1] * 2, shape[1] * 2, shape[2] * 2, dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input[i:i+shape[0], i:i+shape[1], i:i+shape[2]]
    # input = torch.randn(shape[0] * shape[1], shape[1], 2, dtype=dtype, device="cuda")
    input = input.as_strided(shape, (1, shape[0] * shape[2], shape[0]))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, (1, shape[0], shape[0] * shape[1]))

    input_c = input_c[i:i+shape[0], i:i+shape[1], i:i+shape[2]]
    input_c = input_c.as_strided(shape, (1, shape[0] * shape[2], shape[0]))
    output_c = output_c.as_strided(shape, (1, shape[0], shape[0] * shape[1]))

    output_c.copy_(input_c)
    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        print("test_elementwise_copy_3_1_trans_12 Error!")
        exit(1)

def test_elementwise_copy_4_1_trans_12(shape, dtype, i = 1):
    input = torch.randn(shape[0] * 2, shape[1] * 2, shape[2] * 2, shape[3] * 2,  dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input[i:i+shape[0], i:i+shape[1], i:i+shape[2], i:i+shape[3]]
    # input = torch.randn(shape[0] * shape[1], shape[1], 2, dtype=dtype, device="cuda")
    input = input.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

    input_c = input_c[i:i+shape[0], i:i+shape[1], i:i+shape[2], i:i+shape[3]]
    input_c = input_c.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
    output_c = output_c.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

    output_c.copy_(input_c)
    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        print("test_elementwise_copy_4_1_trans_12 Error!")
        exit(1)

if __name__ == "__main__":
    dtypes = [torch.float16, torch.float, torch.bfloat16]
    shapes = [[320, 64], ]
    for dtype in dtypes:
        for shape in shapes:
            test_elementwise_copy_2_1(shape, dtype)

    shapes = [[256, 128, 64], ]
    for dtype in dtypes:
        for shape in shapes:
            test_elementwise_copy_3_1_trans_01(shape, dtype)
            test_elementwise_copy_3_1_trans_012(shape, dtype)
            test_elementwise_copy_3_1_trans_12(shape, dtype)
    
    shapes = [[128, 128, 64, 64], ]
    for dtype in dtypes:
        for shape in shapes:
            test_elementwise_copy_4_1_trans_12(shape, dtype)
    exit(0)
