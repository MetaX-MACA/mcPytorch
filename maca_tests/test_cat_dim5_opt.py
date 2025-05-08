import torch
torch.cuda.manual_seed(0)

dtypes = [torch.float32, torch.float16, torch.bfloat16]
dim = 4

for dtype in dtypes:
    input = torch.rand(*(128,512,20,12,2), dtype=dtype, device='cuda:0')
    input_ = torch.rand(*(128,512,20,12,2), dtype=dtype, device='cuda:0')
    # test cat_for_dim5_copy_non_contiguous_same_inputs_stride
    input1 = torch.as_strided(input, (128,512,20,12,1), (122880,1,6144,512,1,))
    input2 = torch.as_strided(input_, (128,512,20,12,1), (122880,1,6144,512,1,))
    output_g = torch.cat((input1, input2), dim)
    output_cpu = torch.cat((input1.cpu().float(), input2.cpu().float()), dim)
    if not torch.allclose(output_g.cpu().float(), output_cpu):
        exit(1)
    # test cat_for_dim5_copy_non_contiguous_not_same_inputs_stride
    input1 = torch.as_strided(input, (128,512,20,12,1), (122880,1,60,512,1,))
    input2 = torch.as_strided(input_, (128,512,20,12,1), (122880,1,6144,512,1,))
    output_g = torch.cat((input1, input2), dim)
    output_cpu = torch.cat((input1.cpu().float(), input2.cpu().float()), dim)
    if not torch.allclose(output_g.cpu().float(), output_cpu):
        exit(1)
    # test skip opt kernel when last dim != 1
    input1 = torch.rand(*(128,512,20,12,2), dtype=dtype, device='cuda:0')
    input2 = torch.rand(*(128,512,20,12,2), dtype=dtype, device='cuda:0')
    output_g = torch.cat((input1, input2), dim)
    output_cpu = torch.cat((input1.cpu().float(), input2.cpu().float()), dim)
    if not torch.allclose(output_g.cpu().float(), output_cpu):
        exit(1)
    # test cat_for_dim5_copy_contiguous
    input1 = torch.rand(*(121,511,21,11,1), dtype=dtype, device='cuda:0')
    input2 = torch.rand(*(121,511,21,11,1), dtype=dtype, device='cuda:0')
    output_g = torch.cat((input1, input2), dim)
    output_cpu = torch.cat((input1.cpu().float(), input2.cpu().float()), dim)
    if not torch.allclose(output_g.cpu().float(), output_cpu):
        exit(1)