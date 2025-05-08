import torch
import copy

def flip(dtype):
    for s0 in range(1, 200, 23):
        for s1 in range(1, 200, 27):
            shape = [s0, s1]
            a = torch.randn(shape, dtype=dtype)
            a_c = a.as_strided(shape, (0, shape[0]))
            ref = torch.flip(a_c, dims=[0, 1])

            a_d = copy.deepcopy(a).cuda()
            a_d = a_d.as_strided(shape, (0, shape[0]))

            out = torch.flip(a_d, dims=[0, 1])

            if not torch.allclose(ref, out.cpu()):
                print("Error")
                return False
    return True

if __name__ == "__main__":
    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.double]:
        if not flip(dtype):
            print("Error")
            exit(1)
exit(0)