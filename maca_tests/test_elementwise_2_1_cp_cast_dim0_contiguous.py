import torch

def test_2_1_cp_cast_dim0_contiguous():
    inp1_base = torch.rand(100000000,device="cuda",dtype=torch.half)
    out_base = torch.rand(100000000,device="cuda")
    shape_stride_list=[[(3008,4096),(12288,1)],
                       [(1245,128),(344,1)],
                       [(777,256),(104,1)],
                       [(222,64),(112,1)]
                       ]

    for i in range(len(shape_stride_list)):
         shape,stride=shape_stride_list[i]
         out = out_base.as_strided(shape,stride)
         #out = torch.rand(shape,device="cuda")
         inp1 = inp1_base.as_strided(shape,stride)
         out = inp1.float()

         inpc = inp1.cpu()
         outc = inpc.cpu()

         diff= torch.max(out.cpu()-outc)
         if diff > 0.0001:
             print("test_2_1_cp_cast_dim0_contiguous is error")
             exit(1)

test_2_1_cp_cast_dim0_contiguous()
exit(0)

