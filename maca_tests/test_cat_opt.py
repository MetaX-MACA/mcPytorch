import torch
from itertools import product
import copy
import random

shape_list = [[1,2,2,2],[1,2,2,2], [2,2, 3,7], [3,8,16,1], [4,4,16,2],
            [4,4,16,28], [4,40,16,28], [40,40,16,28], [40,40,160,280], 
            [4,4,16,29], [4,40,16,29], [40,40,16,29], [40,40,160,281],
            [1,3,5,7],[17,25,29,23], [22,21, 3,7], [33,67,11,3], [55,99,161,21]]


dtypes = [torch.float16, torch.float, torch.bfloat16]


concat_dim = [0,1,2,3]


for shape, dim, dtype in product(shape_list, concat_dim, dtypes):
    input1 = torch.rand(*shape, dtype=dtype, device='cuda:0')
    new_shape = copy.deepcopy(shape)
    new_shape[dim] = new_shape[dim] // 2 + 1
    input2 = torch.rand(*new_shape, dtype=dtype, device='cuda:0')
    output_g = torch.cat((input1, input2), dim)
    golden = torch.cat((input1.cpu(), input2.cpu()), dim)
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)


#Tests for MACA_CatArrayBatchedCopyAdaptGridSpe
shape_list = [[(256, 8, 720, 720), (256, 8, 720, 1)], [(127, 4, 123, 720), (127, 4, 123, 1)],
              [(33, 27, 721, 720), (33, 27, 721, 1)], [(127, 4, 123, 720), (127, 4, 123, 1)],
              [(256, 8, 720, 1), (256, 8, 720, 720)], [(127, 4, 123, 1), (127, 4, 123, 720)],
              [(33, 27, 721, 1), (33, 27, 721, 720)], [(127, 4, 123, 1), (127, 4, 123, 720)]]
dtypes = [torch.float16, torch.float, torch.bfloat16]


concat_dim = [3]


for shape, dim, dtype in product(shape_list, concat_dim, dtypes):
    input1 = torch.rand(*(shape[0]), dtype=dtype, device='cuda:0')
    input2 = torch.rand(*(shape[1]), dtype=dtype, device='cuda:0')
    output_g = torch.cat((input1, input2), dim)
    golden = torch.cat((input1.cpu(), input2.cpu()), dim)
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)

#Tests for MACA_CatArrayBatchedCopyNoPartialWrite
shape_list = [[(32,112,14,14), (32,56,14,14)], [(32,392,14,14), (32,56,14,14)],
              [(32,280,14,14), (32,56,14,14)], [(32,280,14,14), (32,56,14,14)],
              [(32,78,56,56), (32,26,56,56)], [(32,1456,7,7), (32,208,7,7)],
              [(32,1040,7,7), (32,208,7,7)], [(32,832,7,7), (32,208,7,7)]]
dtypes = [torch.float16, torch.float, torch.bfloat16]


concat_dim = [1]


for shape, dim, dtype in product(shape_list, concat_dim, dtypes):
    input1 = torch.rand(*(shape[0]), dtype=dtype).cuda()
    input2 = torch.rand(*(shape[1]), dtype=dtype).cuda()
    output_g = torch.cat((input1, input2), dim)
    golden = torch.cat((input1.cpu(), input2.cpu()), dim)
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)

#Tests for MACA_CatArrayBatchedCopyNoPartialWrite_3_4_3_0
shape_list = [[(4096,1,32,128), (4096,1,32,128),(4096,1,32,128)]]
dtypes = [torch.float16, torch.float, torch.bfloat16]

concat_dim = [3]

for shape, dim, dtype in product(shape_list, concat_dim, dtypes):
    input1 = torch.rand(*(shape[0]), dtype=dtype, device="cuda")
    input2= torch.rand(*(shape[1]), dtype=dtype, device="cuda")
    input3 = torch.rand(*(shape[2]), dtype=dtype, device="cuda")
    output_g = torch.cat((input1, input2, input3), dim)
    golden = torch.cat((input1.cpu(), input2.cpu(), input3.cpu()), dim)
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)

#Tests for MACA_CatArrayBatchedCopyNoPartialWrite_2_3_2_0
shape_list = [[(4096,1,11008), (4096,1,11008)]]
dtypes = [torch.float16, torch.float, torch.bfloat16]

concat_dim = [2]

for shape, dim, dtype in product(shape_list, concat_dim, dtypes):
    input1 = torch.rand(*(shape[0]), dtype=dtype, device="cuda")
    input2= torch.rand(*(shape[1]), dtype=dtype, device="cuda")
    output_g = torch.cat((input1, input2), dim)
    golden = torch.cat((input1.cpu(), input2.cpu()), dim)
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)

#Tests for input with 0 numel case
shape_list = [[(4096,1,11008), (4096,1,11008)]]
dtypes = [torch.float16, torch.float, torch.bfloat16]

concat_dim = [3]

for shape, dim, dtype in product(shape_list, concat_dim, dtypes):
    input1 = torch.rand(*(4096,1,32,128), dtype=dtype, device='cuda:0')
    input2 = torch.rand(*(4096,1,1,128), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (4096,1,32,128), (4096,4096,128,1))
    input2 = torch.as_strided(input2, (4096,1,32,0), (12288,12288,384,1))
    output_g = torch.cat((input1, input2), dim)
    golden = torch.cat((input1.cpu(), input2.cpu()), dim)
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)

#add test with 0 numel case
for i in range(10):
    input1 = torch.rand(1000).cuda()
    input2 = torch.rand(1000).cuda()
    input3 = torch.rand(182).cuda()
    input4 = torch.rand(0).cuda()
    input5 = torch.rand(2).cuda()
    input6 = torch.rand(0).cuda()

    output_g = torch.cat((input1, input2, input3, input4, input5, input6))
    golden = torch.cat((input1.cpu(), input2.cpu(), input3.cpu(), input4.cpu(), input5.cpu(), input6.cpu()))
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)

#add test with all 0 numel case
for i in range(10):
    input1 = torch.rand(0).cuda()
    input2 = torch.rand(0).cuda()
    input3 = torch.rand(0).cuda()
    input4 = torch.rand(0).cuda()
    input5 = torch.rand(0).cuda()
    input6 = torch.rand(0).cuda()

    output_g = torch.cat((input1, input2, input3, input4, input5, input6))
    golden = torch.cat((input1.cpu(), input2.cpu(), input3.cpu(), input4.cpu(), input5.cpu(), input6.cpu()))
    if not torch.allclose(output_g.cpu(), golden):
        exit(1)

#add traverse test
nDims = [1, 2, 3, 4]
input_length_one_dim = [x for x in range(1, 50, 3)]
dtypes = [torch.float16, torch.float, torch.bfloat16]
num_input_tensors = list(range(1, 130, 5))


for dtype, num_inputs, nDim in product(dtypes, num_input_tensors, nDims):
    shape = [random.choice(input_length_one_dim) for i in range(nDim - 1)]
    for concat_dim in range(nDim):
        input_tensor_list_cuda = []
        input_tensor_list_cpu = []
        for i in range(num_inputs):
            new_shape = copy.deepcopy(shape)
            new_shape.insert(concat_dim, random.choice(input_length_one_dim))
            print(new_shape)
            input = torch.rand(*new_shape, dtype=dtype, device="cuda:0")
            input_tensor_list_cuda.append(input)
            input_tensor_list_cpu.append(input.cpu())
        output_g = torch.cat(input_tensor_list_cuda, dim=concat_dim)
        golden = torch.cat(input_tensor_list_cpu, dim=concat_dim)
        if not torch.allclose(output_g.cpu().float(), golden.float()):
            print(dtype, num_inputs, concat_dim, [x.shape for x in input_tensor_list_cuda])
            exit(1)

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    #add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_non_contiguous_input_last_dim_cat<*,*,1,*,*>
    dim = 0
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (128,), (20,))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (128,), (20,))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (128,), (20,))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopy_noncontig<*,*,1,*,*>
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (129,), (20,))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (129,), (20,))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (129,), (20,))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_non_contiguous_input_last_dim_cat<*,*,2,*,*>
    dim = 1
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (128,12), (20, 1))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (128,12), (20, 1))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (128,12), (20, 1))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)
    
    #add test for MACA_CatArrayBatchedCopy_noncontig<*,*,2,*,*>
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (129,12), (20, 1))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (129,12), (20, 1))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (129,12), (20, 1))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_non_contiguous_input<*,*,2,*,*>
    dim = 0
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (128,12), (20, 1))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (128,12), (20, 1))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (128,12), (20, 1))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_non_contiguous_input_last_dim_cat<*,*,3,*,*>
    dim = 2
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (128,12, 8), (20, 1, 3))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (128,12, 8), (20, 1, 3))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (128,12, 8), (20, 1, 3))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopy_noncontig<*,*,3,*,*>
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (129,12, 8), (20, 1, 3))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (129,12, 8), (20, 1, 3))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (129,12, 8), (20, 1, 3))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_non_contiguous_input<*,*,3,*,*>
    dim = 1
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (128,12, 8), (20, 1, 3))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (128,12, 8), (20, 1, 3))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (128,12, 8), (20, 1, 3))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_non_contiguous_input_last_dim_cat<*,*,4,*,*>
    dim = 3
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (128,12, 8, 4), (20, 1, 3, 2))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (128,12, 8, 4), (20, 1, 3, 2))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (128,12, 8, 4), (20, 1, 3, 2))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopy_noncontig<*,*,4,*,*>
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (129,13, 8, 4), (20, 1, 3, 2))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (129,13, 8, 4), (20, 1, 3, 2))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (129,13, 8, 4), (20, 1, 3, 2))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    #add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_non_contiguous_input<*,*,4,*,*>
    dim = 2
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (128,12, 8, 4), (20, 1, 3, 2))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (128,12, 8, 4), (20, 1, 3, 2))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (128,12, 8, 4), (20, 1, 3, 2))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

#add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_input_not_allsamedtype_last_dim_cat<*,*,1,*,*>
dim = 0
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (128,), (20,))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (128,), (20,))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (128,), (20,))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_input_not_allsamedtype_last_dim_cat<*,*,2,*,*>
dim = 1
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (128,12), (20, 1))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (128,12), (20, 1))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (128,12), (20, 1))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_input_not_allsamedtype<*,*,2,*,*>
dim = 0
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (128,12), (20, 1))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (128,12), (20, 1))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (128,12), (20, 1))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_input_not_allsamedtype_last_dim_cat<*,*,3,*,*>
dim = 2
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (128,12, 8), (20, 1, 3))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (128,12, 8), (20, 1, 3))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (128,12, 8), (20, 1, 3))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_input_not_allsamedtype<*,*,3,*,*>
dim = 1
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (128,12, 8), (20, 1, 3))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (128,12, 8), (20, 1, 3))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (128,12, 8), (20, 1, 3))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_input_not_allsamedtype_last_dim_cat<*,*,4,*,*>
dim = 3
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (128,12, 8, 4), (20, 1, 3, 2))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (128,12, 8, 4), (20, 1, 3, 2))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (128,12, 8, 4), (20, 1, 3, 2))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for MACA_CatArrayBatchedCopyNoPartialWrite_any_any_any_0_input_not_allsamedtype<*,*,4,*,*>
dim = 1
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (128,12, 8, 4), (20, 1, 3, 2))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (128,12, 8, 4), (20, 1, 3, 2))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (128,12, 8, 4), (20, 1, 3, 2))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for copy kernel in cat when input not all same dtype
dim = 1
input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
input1 = torch.as_strided(input1, (37, 11, 8, 4), (20, 1, 3, 2))
input2 = torch.rand(*(248320,120), dtype=torch.float32, device='cuda:0')
input2 = torch.as_strided(input2, (37, 11, 8, 4), (20, 1, 3, 2))
input3 = torch.rand(*(248320,120), dtype=torch.float16, device='cuda:0')
input3 = torch.as_strided(input3, (37, 11, 8, 4), (20, 1, 3, 2))

golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
output_g = torch.cat((input1, input2, input3), dim)
if not torch.allclose(output_g.cpu().float(), golden):
    exit(1)

#add test for MACA_CatArrayBatchedCopy_noncontig when input not all contiguous with ndims in [1, 2, 3, 4]
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    dim = 0
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (3711,), (32,))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (3711,), (32,))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (3711,), (32,))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    dim = 1
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (37, 11), (351, 32))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (37, 11), (351, 32))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (37, 11), (351, 32))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    dim = 1
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (37, 11, 8), (351, 32, 4))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (37, 11, 8), (351, 32, 4))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (37, 11, 8), (351, 32, 4))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    dim = 1
    input1 = torch.rand(*(262272,160), dtype=dtype, device='cuda:0')
    input1 = torch.as_strided(input1, (37, 11, 8, 4), (351, 32, 4, 1))
    input2 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input2 = torch.as_strided(input2, (37, 11, 8, 4), (351, 32, 4, 1))
    input3 = torch.rand(*(248320,120), dtype=dtype, device='cuda:0')
    input3 = torch.as_strided(input3, (37, 11, 8, 4), (351, 32, 4, 1))

    golden = torch.cat((input1.cpu().float(), input2.cpu().float(), input3.cpu().float()) ,dim)
    output_g = torch.cat((input1, input2, input3), dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

#add test for inputs.size() larger than CAT_ARRAY_BATCH_SIZE
num_tensors = [63, 69, 127, 128, 1024]
dim = 1
for num_tensor in num_tensors:
    inputs = [torch.rand(1024, 1).cuda() for i in range(num_tensor)]
    inputs[0] = torch.rand(1024, 1)
    input1 = torch.rand(*(262272,160), dtype=torch.float32, device='cuda:0')
    inputs[0] = torch.as_strided(input1, (1024,1), (8192, 1))
    inputs_cpu = [item.cpu() for item in inputs]
    golden = torch.cat(inputs_cpu, dim)
    output_g = torch.cat(inputs, dim)
    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    inputs = [torch.rand(1024, 1).cuda() for i in range(num_tensor)]
    inputs_cpu = [item.cpu() for item in inputs]
    golden = torch.cat(inputs_cpu, dim)
    output_g = torch.cat(inputs, dim)

    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)

    inputs = [torch.rand(1024, 1).cuda() for i in range(num_tensor)]
    inputs[0] = inputs[0].half()
    inputs_cpu = [item.cpu() for item in inputs]
    golden = torch.cat(inputs_cpu, dim)
    output_g = torch.cat(inputs, dim)

    if not torch.allclose(output_g.cpu().float(), golden):
        exit(1)