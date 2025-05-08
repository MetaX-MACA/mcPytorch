import torch
from itertools import product
import numpy as np
torch.manual_seed(0)
def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=1).table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit = -1))
    prof.export_chrome_trace("test.json")

inp1_base = torch.rand(100000000).cuda()
inp2_base = torch.rand(100000000).cuda()
shape_list2d = [(3,2),(64,128),(128,128),(256,512),(67,127),(138,138),(333,555)]
shape_list3d = [(1,2,3),(128,64,1024),(128,128,32),(128,256,512), (127,61,1024),(125,125,32),(128,255,511)]
shape_list4d = [(3,2,3,1),(32,64,128,256),(31,63,127,254)]

arg_stride_2d=[(33,333),(128,64),(33,88)]
arg_stride_3d=[(33,333,3),(128,64,4),(33,88,9)]
arg_stride_4d=[(33,333,3,3),(128,64,4,32),(33,88,9,129)]

dtype_list = [torch.float,torch.half,torch.bfloat16]


def generate_index(dim,shape):
    shape_dim = shape[dim]
    shape_tmp=list(shape[:])
    shape_tmp[dim]=shape_tmp[-1]
    shape_tmp[-1]=shape_dim
    indexc = torch.zeros(shape_tmp,dtype=torch.int64)
    indexc_tmp=indexc.view(int((np.prod(shape))/shape_tmp[-1]),shape_tmp[-1])
    for i in range(indexc_tmp.size()[0]):
        indexc_tmp[i]=torch.randperm(shape_dim)
    indexc=indexc.transpose(dim,indexc.ndim-1)
    index=indexc.to('cuda')
    return index,indexc


#Tensora ssign 2d
dims=[0,1]
strides=product(arg_stride_2d,arg_stride_2d)
#continuous
for shape in shape_list2d:
    for dim in dims:
        for dtype in dtype_list:
            inp1=torch.rand(shape,device="cuda",dtype=dtype)
            inp2=torch.rand(shape,device="cuda",dtype=dtype)
            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            assert inp1.is_contiguous()
            assert inp2.is_contiguous()
            index,indexc=generate_index(dim,shape)
            inp1.scatter_(dim, index, inp2)
            inp1c.scatter_(dim, indexc, inp2c)
            if not torch.allclose(inp1c, inp1.cpu()):
                print(f"scatter_opt_pw tensor_assign error with shape:{shape}, dtype:{dtype}, dim:{dim},stride:continuous!")
                exit(1)
            print(f"scatter_opt_pw tensor_assign pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
#uncontinuous
for shape in shape_list2d:
    for stride in strides:
        for dim in dims:
            for dtype in dtype_list:
                inp1 = inp1_base.as_strided(shape,stride[0]).to(dtype=dtype)
                inp2 = inp2_base.as_strided(shape,stride[1]).to(dtype=dtype)
                inp1c = inp1.cpu()
                inp2c = inp2.cpu()
                index,indexc=generate_index(dim,shape)
                out=torch.scatter(inp1,dim,index,inp2)
                outc=torch.scatter(inp1c,dim,indexc,inp2c)
                if not torch.allclose(outc, out.cpu()):
                    print(f"scatter_opt_pw tensor_assign error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")
                    exit(1)
                print(f"scatter_opt_pw tensor_assign pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")

#Tensora ssign 3d
dims=[0,1,2]
strides=product(arg_stride_3d,arg_stride_3d)
#continuous
for shape in shape_list3d:
    for dim in dims:
        for dtype in dtype_list:
            inp1=torch.rand(shape,device="cuda",dtype=dtype)
            inp2=torch.rand(shape,device="cuda",dtype=dtype)
            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            assert inp1.is_contiguous()
            assert inp2.is_contiguous()
            index,indexc=generate_index(dim,shape)
            inp1.scatter_(dim, index, inp2)
            inp1c.scatter_(dim, indexc, inp2c)
            if not torch.allclose(inp1c, inp1.cpu()):
                print(f"scatter_opt_pw tensor_assign error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
                exit(1)
            print(f"scatter_opt_pw tensor_assign pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
#uncontinuous
for shape in shape_list3d:
    for stride in strides:
        for dim in dims:
            for dtype in dtype_list:
                inp1 = inp1_base.as_strided(shape,stride[0]).to(dtype=dtype)
                inp2 = inp2_base.as_strided(shape,stride[1]).to(dtype=dtype)
                inp1c = inp1.cpu()
                inp2c = inp2.cpu()
                index,indexc=generate_index(dim,shape)
                out=torch.scatter(inp1,dim,index,inp2)
                outc=torch.scatter(inp1c,dim,indexc,inp2c)
                if not torch.allclose(outc, out.cpu()):
                    print(f"scatter_opt_pw tensor_assign error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")
                    exit(1)
                print(f"scatter_opt_pw tensor_assign pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")
#Tensora ssign 4d
dims=[0,1,2,3]
strides=product(arg_stride_4d,arg_stride_4d)
#continuous
for shape in shape_list4d:
    for dim in dims:
        for dtype in dtype_list:
            inp1=torch.rand(shape,device="cuda",dtype=dtype)
            inp2=torch.rand(shape,device="cuda",dtype=dtype)
            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            assert inp1.is_contiguous()
            assert inp2.is_contiguous()
            index,indexc=generate_index(dim,shape)
            inp1.scatter_(dim, index, inp2)
            inp1c.scatter_(dim, indexc, inp2c)
            if not torch.allclose(inp1c, inp1.cpu()):
                print(f"scatter_opt_pw tensor_assign error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
                exit(1)
            print(f"scatter_opt_pw tensor_assign pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
#uncontinuous
for shape in shape_list4d:
    for stride in strides:
        for dim in dims:
            for dtype in dtype_list:
                inp1 = inp1_base.as_strided(shape,stride[0]).to(dtype=dtype)
                inp2 = inp2_base.as_strided(shape,stride[1]).to(dtype=dtype)
                inp1c = inp1.cpu()
                inp2c = inp2.cpu()
                index,indexc=generate_index(dim,shape)
                out=torch.scatter(inp1,dim,index,inp2)
                outc=torch.scatter(inp1c,dim,indexc,inp2c)
                if not torch.allclose(outc, out.cpu()):
                    print(f"scatter_opt_pw tensor_assign error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")
                    exit(1)
                print(f"scatter_opt_pw tensor_assign pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")


#Reduce add 2d
dims=[0,1]
strides=product(arg_stride_2d,arg_stride_2d)
#continuous
for shape in shape_list2d:
    for dim in dims:
        for dtype in dtype_list:
            inp1=torch.rand(shape,device="cuda",dtype=dtype)
            inp2=torch.rand(shape,device="cuda",dtype=dtype)
            index = torch.randint(0, shape[dim], shape, device="cuda")
            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            indexc=index.cpu()
            assert inp1.is_contiguous()
            assert inp2.is_contiguous()
            inp1.scatter_add_(dim, index, inp2)
            inp1c.scatter_add_(dim, indexc, inp2c)
            if dtype is torch.half:
                check=torch.allclose(inp1c, inp1.cpu(),rtol=1e-2,atol=1e-4)
            elif dtype is torch.bfloat16:
                check=torch.allclose(inp1c, inp1.cpu(),rtol=1e-1,atol=1e-2)
            else:
                check=torch.allclose(inp1c, inp1.cpu())
            if not check:
                print(f"scatter_opt_pw reduce_add error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
                exit(1)
            print(f"scatter_opt_pw reduce_add pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
#uncontinuous
for shape in shape_list2d:
    for stride in strides:
        for dim in dims:
            for dtype in dtype_list:
                index = torch.randint(0, shape[dim], shape, device="cuda")
                inp1 = inp1_base.as_strided(shape,stride[0]).to(dtype=dtype)
                inp2 = inp2_base.as_strided(shape,stride[1]).to(dtype=dtype)
                indexc=index.cpu()
                inp1c = inp1.cpu()
                inp2c = inp2.cpu()
                out=torch.scatter(inp1,dim,index,inp2,reduce='add')
                outc=torch.scatter(inp1c,dim,indexc,inp2c,reduce='add')
                if dtype is torch.half:
                    check=torch.allclose(outc, out.cpu(),rtol=1e-2,atol=1e-4)
                elif dtype is torch.bfloat16:
                    check=torch.allclose(outc, out.cpu(),rtol=1e-1,atol=1e-2)
                else:
                    check=torch.allclose(outc, out.cpu())
                if not check:
                    print(f"scatter_opt_pw reduce_add error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")
                    exit(1)
                print(f"scatter_opt_pw reduce_add pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")

#Reduce add 3d
dims=[0,1,2]
strides=product(arg_stride_3d,arg_stride_3d)
#continuous
for shape in shape_list3d:
    for dim in dims:
        for dtype in dtype_list:
            inp1=torch.rand(shape,device="cuda",dtype=dtype)
            inp2=torch.rand(shape,device="cuda",dtype=dtype)
            index = torch.randint(0, shape[dim], shape, device="cuda")
            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            indexc=index.cpu()
            assert inp1.is_contiguous()
            assert inp2.is_contiguous()
            inp1.scatter_add_(dim, index, inp2)
            inp1c.scatter_add_(dim, indexc, inp2c)
            if dtype is torch.half:
                check=torch.allclose(inp1c, inp1.cpu(),rtol=1e-2,atol=1e-4)
            elif dtype is torch.bfloat16:
                check=torch.allclose(inp1c, inp1.cpu(),rtol=1e-1,atol=1e-2)
            else:
                check=torch.allclose(inp1c, inp1.cpu())
            if not check:
                print(f"scatter_opt_pw reduce_add error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
                exit(1)
            print(f"scatter_opt_pw pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
#uncontinuous
for shape in shape_list3d:
    for stride in strides:
        for dim in dims:
            for dtype in dtype_list:
                index = torch.randint(0, shape[dim], shape, device="cuda")
                inp1 = inp1_base.as_strided(shape,stride[0]).to(dtype=dtype)
                inp2 = inp2_base.as_strided(shape,stride[1]).to(dtype=dtype)
                indexc=index.cpu()
                inp1c = inp1.cpu()
                inp2c = inp2.cpu()
                out=torch.scatter(inp1,dim,index,inp2,reduce='add')
                outc=torch.scatter(inp1c,dim,indexc,inp2c,reduce='add')
                if dtype is torch.half:
                    check=torch.allclose(outc, out.cpu(),rtol=1e-2,atol=1e-4)
                elif dtype is torch.bfloat16:
                    check=torch.allclose(outc, out.cpu(),rtol=1e-1,atol=1e-2)
                else:
                    check=torch.allclose(outc, out.cpu())
                if not check:
                    print(f"scatter_opt_pw reduce_add error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")
                    exit(1)
            print(f"scatter_opt_pw reduce_add pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")



#Reduce add 4d
dims=[0,1,2,3]
strides=product(arg_stride_4d,arg_stride_4d)
#continuous
for shape in shape_list4d:
    for dim in dims:
        for dtype in dtype_list:
            inp1=torch.rand(shape,device="cuda",dtype=dtype)
            inp2=torch.rand(shape,device="cuda",dtype=dtype)
            index = torch.randint(0, shape[dim], shape, device="cuda")
            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            indexc=index.cpu()
            assert inp1.is_contiguous()
            assert inp2.is_contiguous()
            inp1.scatter_add_(dim, index, inp2)
            inp1c.scatter_add_(dim, indexc, inp2c)
            if dtype is torch.half:
                check=torch.allclose(inp1c, inp1.cpu(),rtol=1e-2,atol=1e-4)
            elif dtype is torch.bfloat16:
                check=torch.allclose(inp1c, inp1.cpu(),rtol=1e-1,atol=1e-2)
            else:
                check=torch.allclose(inp1c, inp1.cpu())
            if not check:
                print(f"scatter_opt_pw reduce_add error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
                exit(1)
            print(f"scatter_opt_pw reduce_add pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride:continuous!")
#uncontinuous
for shape in shape_list4d:
    for stride in strides:
        for dim in dims:
            for dtype in dtype_list:
                index = torch.randint(0, shape[dim], shape, device="cuda")
                inp1 = inp1_base.as_strided(shape,stride[0]).to(dtype=dtype)
                inp2 = inp2_base.as_strided(shape,stride[1]).to(dtype=dtype)
                indexc=index.cpu()
                inp1c = inp1.cpu()
                inp2c = inp2.cpu()
                out=torch.scatter(inp1,dim,index,inp2,reduce='add')
                outc=torch.scatter(inp1c,dim,indexc,inp2c,reduce='add')
                if dtype is torch.half:
                    check=torch.allclose(outc, out.cpu(),rtol=1e-2,atol=1e-4)
                elif dtype is torch.bfloat16:
                    check=torch.allclose(outc, out.cpu(),rtol=1e-1,atol=1e-2)
                else:
                    check=torch.allclose(outc, out.cpu())
                if not check:
                    print(f"scatter_opt_pw reduce_add error with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")
                    exit(1)
                print(f"scatter_opt_pw reduce_add pass with shape:{shape}, dtype:{dtype}, dim:{dim}, stride0:{stride[0]},stride1:{stride[1]}!")



with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU],
    on_trace_ready=trace_handler, record_shapes = True) as prof:
    shape=[784,256,128]
    dtype=torch.float
    dim=0
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    index = torch.randint(0, shape[dim], shape, device="cuda")
    for i in range(10):
        out=torch.scatter(inp1,dim,index,inp2,reduce='add')

exit(0)