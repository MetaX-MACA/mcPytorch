import torch

def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=1).table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit = -1))
    prof.export_chrome_trace("test.json")



inp1_base = torch.rand(100000000).cuda()
inp2_base = torch.rand(100000000).cuda()

shape_list = [(128,64,3136),(128,128,784),(128,320,196),(128,512,64),
              (128,96,3131),(128,192,784),(128,384,196),(128,768,64)]

arg1_stride=[(33,333,2),(124,24,3),(33,88,9)]
arg2_stride=[(3,13,22),(17,7,22),(135,22,11),(11,33,44)]

dtype_list = [torch.float,torch.half,torch.bfloat16]


for shape in shape_list:
    for stride1 in arg1_stride:
        for stride2 in arg2_stride:
            for dtype1 in dtype_list:
                for dtype2 in dtype_list:

                    inp1 = inp1_base.as_strided(shape,stride1).to(dtype=dtype1)
                    inp2 = inp2_base.as_strided(shape,stride2).to(dtype=dtype2)
                    out = inp1 + inp2

                    inp1c = inp1.cpu()
                    inp2c = inp2.cpu()
                    outc = inp1c + inp2c

                    diff = torch.max(torch.abs(out.cpu()-outc))
                    if diff > 0.0001:
                        print("test_ele32_opt_tile.py is error")
                        exit(1)

# shape form model
shape_stride_list = [[[2048,4,60416], [60416*4,60416,1], [1,2048,0]], [[1024,5,60416], [302080,60416,1], [1,2048,0]],
                    [[6147,64,80], [5120,80,1], [80,0,1]], [[2048, 5, 1920], [9600, 1920, 1], [1920, 3932160, 1]]]

for dtype in dtype_list:
    for shape, stride_out, stride_in in shape_stride_list:
        inp1_base = torch.rand(1000000000,device="cuda",dtype=dtype)
        inp2_base = torch.rand(1000000000,device="cuda",dtype=dtype)

        inp1_base_c = inp1_base.detach().clone().cpu()
        inp2_base_c = inp2_base.detach().clone().cpu()

        inp1 = inp1_base.as_strided(shape, stride_out)
        inp2 = inp2_base.as_strided(shape, stride_in)

        inpc1 = inp1_base_c.as_strided(shape, stride_out)
        inpc2 = inp2_base_c.as_strided(shape, stride_in)

        out = inp1 + inp2
        outc = inpc1 + inpc2
        res=torch.allclose(out.cpu(), outc)
        if not res:
            print("test_ele32_opt_tile ERROR!")
            exit(1)
def test_bug():
    dtype=torch.float32
    device='cuda'
    t = torch.rand(
                    (2048,2,64,128),
                    dtype=dtype,
                    device=device,
                )
    t.requires_grad = True
    emb = torch.rand(
        (2048, 1, 1, 128),
        dtype=dtype,
        device=device,
    )
    out=t*emb
    loss=out.sum()*2
    loss.backward()
test_bug()
exit(0)
