import torch
import copy

def test_elementwise_3_2_cast_broadcast_dim2(shapes, dtype1, dtype2):

    for shape in shapes:
        a = torch.randn(shape, dtype=dtype1, device="cuda")
        b = torch.randn(shape, dtype=dtype2, device="cuda")
        b = b.as_strided(shape, (0, 1, 0))

        # test correctness
        c = a + b
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        c_cpu = a_cpu + b_cpu

        if not torch.allclose(c.cpu(), c_cpu):
            return False
    return True

if __name__ == "__main__":
    shapes = [[128, 8, 2304], [128, 2048, 64], [64, 2047, 128], [68, 33, 192], [162, 33, 128], [63, 2048, 128], [7, 2, 128], [33, 49, 128], [257, 127, 128]]

    if not test_elementwise_3_2_cast_broadcast_dim2(shapes, torch.float16, torch.float32):
        print("elementwise_3_2_cast_broadcast_dim2 Error!!!")
        exit(1)
    
    exit(0)