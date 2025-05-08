import torch

lines_list = ["ee,2,1,128,256,2,256,512,2",
              "ee,2,1,65536,128,2,131072,512,2",
              "ee,2,1,128,65536,2,512,131072,2",
              "ee,2,1,64,65536,2,512,44444,2",
              "ee,2,1,64,64,2,512,128,2",
              "ee,2,1,64,128,2,128,512,2",
              "ee,2,1,64,1024,2,128,4096,2",
              "ee,3,1,128,256,256,2,512,131072,131072,512,2",
              "ee,3,1,64,128,512,2,256,131072,262144,1024,2",
              "ee,3,1,64,64,64,2,128,16384,16384,128,2",
              "ee,3,1,128,64,128,2,512,32768,32768,256,2",
              "ee,3,1,128,11,128,2,512,32768,32768,256,2",
              "ee,3,1,128,11,256,2,512,32768,22528,1024,2"]

def test_elementwise_kernel_transpose_copy_64_uncontiguous():
    for line in lines_list:

        line_list = line.split(",")
        dim = int(line_list[1])
        arity = int(line_list[2])

        shape = []
        for i in range(dim):
            shape.insert(0, int(line_list[3+i]))
        out_stride = []
        for i in range(dim):
            out_stride.insert(0, int(line_list[3+dim+i]))
        inp_stride = []
        for i in range(dim):
            inp_stride.insert(0, int(line_list[3+dim*2+i]))


        dtype=torch.float16
        if out_stride[dim-1] == 4:
            dtype=torch.float32
        if out_stride[dim-1]!=2 and out_stride[dim-1]!=4:
            exit(1)

        for i in range(dim):
            inp_stride[i] = int(inp_stride[i] / out_stride[dim-1])
            out_stride[i] = int(out_stride[i] / out_stride[dim-1])

        inp_base = torch.rand(1500000000,device="cuda",dtype=dtype)
        out_base = torch.rand(1500000000,device="cuda",dtype=dtype)

        inp = inp_base.as_strided(shape, inp_stride)
        out = out_base.as_strided(shape, out_stride)

        out.copy_(inp)
        diff = torch.max(torch.abs(out-inp)).cpu()
        if diff > 0.00001:
            print(diff)
            print(line)
            print("test_elementwise_kernel_transpose_copy_64_uncontiguous is error")
            exit(1)


test_elementwise_kernel_transpose_copy_64_uncontiguous()

