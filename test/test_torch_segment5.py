import torch
import unittest
from common_util import *


class TestTorchFunctionSegment5(unittest.TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_cross_backward(self, device, dtype):
        shapes = [(2, 3), (4, 6, 3), (20, 3, 20, 4, 5)]
        dims = [1, 2, 1]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        others = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            other = others[i]
            dim = dims[i]

            input_list = [input, other, dim]
            if dtype in [torch.half]:
                tol = 1e-2
            
            runtestapi(func=torch.cross, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)

    
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_cummax_backward(self, device, dtype):
        a = torch.tensor([-0.3449, -1.5447, 0.0685, -1.5104, -1.1706, 0.2259,
                      1.4696, -1.3284, 1.9946, -0.8209], device = device)
        a.requires_grad = True
        out = torch.cummax(a, dim = 0)
        values = out[0]
        values_expect = torch.tensor([-0.3449, -0.3449, 0.0685, 0.0685, 0.0685,
                                    0.2259, 1.4696, 1.4696, 1.9946, 1.9946], device = device)
        indices = out[1]
        indices_expect = torch.tensor([0, 0, 2, 2, 2, 5, 6, 6, 8, 8], device = device)

        values.backward(torch.ones(values.shape).cuda())
        grad_expect = torch.tensor([2., 0., 3., 0., 0., 1., 2., 0., 2., 0.], device= device)

        assert torch.allclose(indices, indices_expect)
        assert torch.allclose(values, values_expect)
        assert torch.allclose(a.grad, grad_expect)
    

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_cummin_backward(self, device, dtype):
        a = torch.tensor([-0.2517,  0.9014,  0.3438,  1.6492, -0.0656,  0.2198,  
                       0.3921,  0.5053,  1.5771, -2.7968], device = device)
        a.requires_grad = True
        out = torch.cummin(a, dim = 0)
        values = out[0]
        values_expect = torch.tensor([-0.2517, -0.2517, -0.2517, -0.2517, -0.2517, -0.2517,
                                    -0.2517, -0.2517, -0.2517, -2.7968], device=device)
        indices = out[1]
        indices_expect = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 9], device=device)
        values.backward(torch.ones(values.shape).cuda())
        grad_expect = torch.tensor([9., 0., 0., 0., 0., 0., 0., 0., 0., 1.], device=device)

        assert torch.allclose(indices, indices_expect)
        assert torch.allclose(values, values_expect)
        assert torch.allclose(a.grad, grad_expect)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_cummin_cummax(self, device, dtype):
        def test_help(func, shape, dim):
            inputs = gendata([shape], type_list=[dtype], rand_algo="rand")
            tol = 1e-5

            if dtype in [torch.half,torch.bfloat16]:
                input_c = inputs[0].detach().float()
                tol = 1e-2
            else:
                input_c = inputs[0].detach()
            input_g = inputs[0].detach().cuda()
            input_c.requires_grad = True
            input_g.requires_grad = True

            out = func(input_c, dim = dim)
            backward_input_c = torch.rand(out[0].shape).to(dtype=dtype)
            backward_input_g = backward_input_c.detach().clone().cuda()
            if dtype in [torch.half]:
                backward_input_c = backward_input_c.float()
            out[0].backward(backward_input_c)

            out1 = op(input_g, dim = dim)
            out1[0].backward(backward_input_g)

            if dtype not in [torch.half,torch.bfloat16]:
                assert torch.allclose(out1[0].cpu(), out[0], rtol=tol, atol=tol)
                assert torch.allclose(out1[1].cpu(), out[1], rtol=tol, atol=tol)
                assert torch.allclose(input_g.grad.cpu(), input_c.grad, rtol=tol, atol=tol)
            else:
                assert torch.allclose(out1[0].float().cpu(), out[0], rtol=tol, atol=tol)
                assert torch.allclose(out1[1].cpu(), out[1], rtol=tol, atol=tol)
                assert torch.allclose(input_g.grad.float().cpu(), input_c.grad, rtol=tol, atol=tol)

        ops = [torch.cummax, torch.cummin]
        for op in ops:
            test_help(op, (2, 3), 0)
            test_help(op, (3, 4, 5), 1)
            test_help(op, (4, 5, 6, 7), 1)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_cumprod(self, device, dtype):
        a = torch.tensor([0.6776, 0.1292, 0.0092, 0.8378, 0.8770, 0.3592, 
                          0.0014, 0.8356, 0.9043, 0.0947], device = device, dtype=dtype)
        out = torch.cumprod(a, dim = 0)
        out_expect = torch.tensor([6.7760e-01, 8.7546e-02, 8.0542e-04, 6.7478e-04, 5.9178e-04, 2.1257e-04,
                                2.9760e-07, 2.4867e-07, 2.2487e-07, 2.1296e-08], device= device, dtype=dtype)
        a[5] = 0.0
        out1 = torch.cumprod(a, dim = 0)
        out1_expect = torch.tensor([6.7760e-01, 8.7546e-02, 8.0542e-04, 6.7478e-04, 5.9178e-04, 0.0000e+00,
                                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device= device, dtype=dtype)
        
        assert torch.allclose(out, out_expect)
        assert torch.allclose(out1, out1_expect)
    

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_cumprod_backward(self, device, dtype):
        a = torch.tensor([0.6776, 0.1292, 0.0092, 0.8378, 0.8770, 0.3592, 
                      0.0014, 0.8356, 0.9043, 0.0947], device= device, dtype=dtype)
        a.requires_grad = True

        out = torch.cumprod(a, dim = 0)
        out.backward(torch.ones(out.shape).cuda())
        grad_expect = torch.tensor([1.1326e+00, 6.9529e-01, 2.4841e-01, 1.7664e-03, 9.1807e-04, 5.9399e-04,
                                    5.6603e-04, 5.9220e-07, 2.7222e-07, 2.2487e-07], device= device, dtype=dtype)
        rtol, atol=1e-4, 1e-4
        if dtype is torch.half:
            rtol=1e-3
            atol=1e-3
        assert torch.allclose(a.grad, grad_expect, rtol=rtol, atol=atol)
    

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_cumprod_unconting(self, device, dtype):
        shapes = [(5), (2, 3), (3, 4, 5), (2,3,4,3)]
        dims = [0, 0, 2, 1]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            dim = dims[i]
            input = inputs[i]
            input_list = [input, dim]
            if dtype in [torch.half]:
                tol = 1e-2

            runtestapi(func=torch.cumprod, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_cumsum_backward(self, device, dtype):
        a = torch.tensor([0.5189, 0.9599, 0.6774, 0.8756, 0.5053, 0.4253, 
                      0.8884, 0.3479, 0.3208, 0.4868], device=device, dtype=dtype)
        a.requires_grad = True
        out = torch.cumsum(a, dim = 0)
        out.backward(torch.ones(out.shape).cuda())
        grad_expect = torch.tensor([10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.], device=device, dtype=dtype)

        assert torch.allclose(a.grad, grad_expect)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_cumsum_unconting(self, device, dtype):
        shapes = [(5), (2, 3), (3, 4, 5), (2,3,4,3)]
        dims = [0, 0, 2, 1]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            dim = dims[i]
            input_list = [input, dim]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.cumsum, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False))) 
    def test_diag(self, device, dtype):
        shapes = [(3), (5), (10), (4, 6), (3, 3), (11, 12), (40, 50)]
        diagonals = [0, 1, -2, -2, 1, -6, 10]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            diagonal = diagonals[i]
            input_list = [input, diagonal]
            if dtype in [torch.half, torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.diag, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_diag_embed(self, device, dtype):
        shapes = [(2,3,4), (5,5), (6), (5,6,7,8), (21,40,8,20,8)]
        offsets = [1, 2, 0, 1, 3]
        dims = [(0,2),(0,1), (0,1),(1,3),(3,4)]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            offset = offsets[i]
            dim1, dim2 = dims[i]
            input_list = [input, offset, dim1, dim2]
            if dtype in [torch.half, torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.diag_embed, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_diagflat(self, device, dtype):
        shapes = [(2,3,4), (5,5), (6), (5,6,7,8)]
        offsets = [1, 2, 0, 1]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            offset = offsets[i]
            input_list = [input, offset]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.diagflat, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_diagonal(self, device, dtype):
        shapes = [(2,3,4), (5,5), (5,6,7,8), (100,200,300)]
        offsets = [4, -2, 0, -20]
        dims = [(2,0),(0,1),(2,3),(2,0)]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            offset = offsets[i]
            dim1, dim2 = dims[i]
            input_list = [input, offset, dim1, dim2]
            if dtype in [torch.half]:
                tol = 1e-3

            runtestapi(func=torch.diagonal, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_diff(self, device, dtype):
        shapes = [(4), (3,4), (5,6,7), (8,9,10,11), (2,2,2,2,2,2)]
        pendshapes = [(4), (2,4), (5,5,7), (8,9,4,11), (2,2,2,2,2,2)]
        trans_shapes = [(0), (0,1), (2,1,0), (0,2,3,1), (5,4,3,2,1,0)]
        dims = (0, 0, 1, 1, 5)

        for i in range(len(shapes)):
            shape = shapes[i]
            pendshape = pendshapes[i]
            trans_shape = trans_shapes[i]
            dim = dims[i]
            inp1 = torch.rand(shape).permute(trans_shape).to(dtype=dtype)
            inp2 = torch.rand(pendshape).permute(trans_shape).to(dtype=dtype)

            cpu_dytpe = dtype
            if dtype is torch.half:
                 cpu_dytpe = torch.float
            
            a = inp1.detach().to(dtype=cpu_dytpe)
            b = inp2.detach().to(dtype=cpu_dytpe)
            c = inp2.detach().to(dtype=cpu_dytpe)
            a.requires_grad = True
            b.requires_grad = True
            c.requires_grad = True
            out = torch.diff(a, dim=dim, prepend=b, append=c)
            out.backward(torch.ones(out.shape).to(dtype=cpu_dytpe))

            a1 = inp1.detach().to(dtype=dtype)
            b1 = inp2.detach().to(dtype=dtype)
            c1 = inp2.detach().to(dtype=dtype)
            a1.requires_grad = True
            b1.requires_grad = True
            c1.requires_grad = True
            out1 = torch.diff(a1, dim=dim, prepend=b1, append=c1)
            out1.backward(torch.ones(out1.shape).to(dtype=dtype))

            assert out1.size() == out.size()
            assert out1.stride() == out.stride()
            if dtype is not torch.half:
                assert torch.allclose(out1.cpu(), out)
                assert torch.allclose(a1.grad.cpu(), a.grad)
                assert torch.allclose(b1.grad.cpu(), b.grad)
                assert torch.allclose(c1.grad.cpu(), c.grad)
            else:
                assert torch.allclose(out1.float().cpu(), out, rtol=1e-3, atol=1e-3)
                assert torch.allclose(a1.grad.float().cpu(), a.grad, rtol=1e-3, atol=1e-3)
                assert torch.allclose(b1.grad.float().cpu(), b.grad, rtol=1e-3, atol=1e-3)
                assert torch.allclose(c1.grad.float().cpu(), c.grad, rtol=1e-3, atol=1e-3)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_flatten(self, device, dtype):
        shapes = [(2,3,4), (5,5), (5,6,7,8), (100,200,300), (3,4,5,6,7), (1)]
        dims = [(0,1), (1,-1), (1,2),(0,-1),(2,4),(0,0)]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            dim1,dim2 = dims[i]
            input_list = [input, dim1, dim2]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2

            runtestapi(func=torch.flatten, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_flip(self, device, dtype):
        shapes = [(1,2,3), (2,4,6), (3,4), (4,5,6,7,8), (1,2,3,4,5,6,7)]
        dims = [(0,1),(1,2),(0,1),(0,1,2,3,4),(2,3,5)]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            dim = dims[i]
            input_list = [input,dim]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2

            runtestapi(func=torch.flip, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_fliplr(self, device, dtype):
        shapes = [(1,2,3), (2,4,6), (3,4), (4,5,6,7,8), (1,2,3,4,5,6,7)]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input_list = [inputs[i]]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.fliplr, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_flipud(self, device, dtype):
        shapes = [(1,2,3), (2,4,1), (3,3), (4,5,2,7,8), (10,4,5,6,7)]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input_list = [inputs[i]]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.flipud, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)



    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_kron(self, device, dtype):
        shapes1 = [(3), (2,3), (3,4), (10,10,10), (3,4,5,6,7)]
        shapes2 = [(3), (2,3), (3,4,9), (10,10,10), (3,4,5,6,7)]

        inputs1 = gendata(shapes1, type_list=[dtype], rand_algo="rand")
        inputs2 = gendata(shapes2, type_list=[dtype], rand_algo="rand")
        tol = 1e-3
        for input1 in inputs1:
            for input2 in inputs2:
                input_list = [input1, input2]
                if dtype in [torch.half,torch.bfloat16]:
                    tol = 1e-2
                
                runtestapi(func=torch.kron, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                        low=0, high=1, compare_type="torch")


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_rot90(self, device, dtype):
        shapes = [(3,3,3), (2,4,7), (2,9), (4,5,2,7,8), (1,4,5,6,7)]
        times = [1, 2, 3, 4, 5]
        dims = [[0,1], [1,2], [1,0], [3,4],[1,4]]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            time = times[i]
            dim = dims[i]
            input_list = [input, time, dim]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2

            runtestapi(func=torch.rot90, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*(torch.int32, torch.int64))
    def test_gcd(self, device, dtype):
        shapes = [(2,), (3,5), (6,10,20,30), (9, 10), (34, 22)]
        for shape in shapes:
            input1 = make_tensor(shape, dtype=dtype, device='cpu', low=10, high=1000, noncontiguous=True)
            input2 = make_tensor(shape, dtype=dtype, device='cpu', low=10, high=1000, noncontiguous=True)
            input_c = [input1, input2]
            input_g = [input1.detach().clone().cuda(), input2.detach().clone().cuda()]

            out = torch.gcd(*input_c)
            out1 = torch.gcd(*input_g)

            assert out1.size() == out.size()
            assert out1.stride() == out.stride()
            assert torch.allclose(out1.cpu(), out)



    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_histc(self, device, dtype):
        shapes = [(10,20,30), (20,40,10), (30,30), (40,50,20,70,80), (10,4,50,6,70)]
        trans_shapes = [(0,2,1), (0,1,2),(0,1),(0,1,2,3,4),(1,0,3,2,4)]
        args = ([5, 20], [20,70], [15, 80], [14, 90], [3, 20])

        for i in range(len(shapes)):
            shape = shapes[i]
            trans_shape = trans_shapes[i]
            bin, ran = args[i]
            inp1 = (torch.rand(shape).permute(trans_shape).to(dtype=dtype) - 0.5) * ran * 2
            a = inp1.detach()
            out = torch.histc(a, bins=bin, min=-ran, max=ran)
            a1 = inp1.detach().cuda()
            out1 = torch.histc(a1, bins=bin, min=-ran, max=ran)

            assert out1.size() == out.size()
            assert out1.stride() == out.stride()
            assert torch.allclose(out1.cpu(), out)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_meshgrid_backward(self, device, dtype):
        shapes1 = [(10,),(11,)]
        shapes2 = [(5,),(8,)]
        shapes3 = [(21,),(22,)]
        indexs = ["ij", "xy"]
        
        for shape1 in shapes1:
            for shape2 in shapes2:
                for shape3 in shapes3:
                    for index in indexs:
                        cpu_dytpe = dtype
                        if dtype in [torch.half,torch.bfloat16]:
                            cpu_dytpe = torch.float

                        a = torch.rand(shape1).to(dtype=cpu_dytpe)
                        b = torch.rand(shape2).to(dtype=cpu_dytpe)
                        c = torch.rand(shape3).to(dtype=cpu_dytpe)
                        a.requires_grad = True
                        b.requires_grad = True
                        c.requires_grad = True
                        oc1, oc2, oc3 = torch.meshgrid(a, b, c, indexing=index)
                        oc1.backward(torch.ones(oc1.shape).to(dtype=cpu_dytpe))
                        oc2.backward(torch.ones(oc2.shape).to(dtype=cpu_dytpe))
                        oc3.backward(torch.ones(oc3.shape).to(dtype=cpu_dytpe))

                        a1 = a.detach().clone().cuda().to(dtype=dtype)
                        b1 = b.detach().clone().cuda().to(dtype=dtype)
                        c1 = c.detach().clone().cuda().to(dtype=dtype)
                        a1.requires_grad = True
                        b1.requires_grad = True
                        c1.requires_grad = True
                        og1, og2, og3 = torch.meshgrid(a1, b1, c1, indexing=index)
                        og1.backward(torch.ones(og1.shape).cuda().to(dtype=dtype))
                        og2.backward(torch.ones(og2.shape).cuda().to(dtype=dtype))
                        og3.backward(torch.ones(og3.shape).cuda().to(dtype=dtype))

                        if dtype not in [torch.half,torch.bfloat16]:
                            assert torch.allclose(og1.cpu(), oc1)
                            assert torch.allclose(og2.cpu(), oc2)
                            assert torch.allclose(og3.cpu(), oc3)
                            assert torch.allclose(a1.grad.cpu(), a.grad)
                            assert torch.allclose(b1.grad.cpu(), b.grad)
                            assert torch.allclose(c1.grad.cpu(), c.grad)
                        else:
                            assert torch.allclose(og1.float().cpu(), oc1, rtol=1e-2, atol=1e-2)
                            assert torch.allclose(og2.float().cpu(), oc2, rtol=1e-2, atol=1e-2)
                            assert torch.allclose(og3.float().cpu(), oc3, rtol=1e-2, atol=1e-2)
                            assert torch.allclose(a1.grad.float().cpu(), a.grad, rtol=1e-2, atol=1e-2)
                            assert torch.allclose(b1.grad.float().cpu(), b.grad, rtol=1e-2, atol=1e-2)
                            assert torch.allclose(c1.grad.float().cpu(), c.grad, rtol=1e-2, atol=1e-2)


    @onlyCUDA
    @dtypesIfCUDA(*(torch.int32, torch.int64))
    def test_lcm(self, device, dtype):   
        shapes = [(10,), (23,24), (10,11,12), (28,29,20,21)]
        ranges = [(1,10), (20, 40)]
        tol = 1e-5
        for shape in shapes:
            for range in ranges:
                input1 = make_tensor(shape, dtype=dtype, device='cpu', low=range[0], high=range[1], noncontiguous=True)
                input2 = make_tensor(shape, dtype=dtype, device='cpu', low=range[0], high=range[1], noncontiguous=True)
                input_c = [input1, input2]
                input_g = [input1.detach().clone().cuda(), input2.detach().clone().cuda()]

                out = torch.lcm(*input_c)
                out1 = torch.lcm(*input_g)

                assert out1.size() == out.size()
                assert out1.stride() == out.stride()
                assert torch.allclose(out1.cpu(), out)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_logcumsumexp(self, device, dtype):
        shapes = [(1,2,3), (2,4,1), (3,3), (4,5,2,7,8), (1,4,5,6,7)]
        dims = [0, 1, 1, 2, 3]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            dim = dims[i]
            input_list = [input, dim]

            runtestapi(func=torch.logcumsumexp, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_ravel(self, device, dtype):
        shapes = [(10,20),(10),(23,11,24),(14,47,10,7),((2,3,4,5,6,7))]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol = 1e-5
        for i in range(len(inputs)):
            input_list = [inputs[i]]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.ravel, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_renorm(self, device, dtype):
        shapes = [(10,20),(10,2),(23,11,24)]
        dims = [0, 0, 2]
        ps = [0.2, 1.]
        maxnorms = [0.3, 2.5]

        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol=1e-5
        for i in range(len(inputs)):
            input = inputs[i]
            for p in ps:
                for maxnorm in maxnorms:  
                    input_list = [input, p, dims[i], maxnorm]

                    runtestapi(func=torch.renorm, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_roll(self, device, dtype):
        shapes = [(2,3),(5,6),(7,8,9),(11,12,13,14),(5,6,7,8,9,10,11)]
        shiftss = [(2,3),(2),(2,-1),(-1,-2,-2),(2,-1,-2,3,-4,4,5)]
        dimss = [(0,1),(0),(0,1),(0,2,3),(0,1,2,3,4,5,6)]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol=1e-5

        for i in range(len(inputs)):
            input = inputs[i]
            shifts = shiftss[i]
            dims = dimss[i]
            input_list = [input, shifts, dims]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.roll, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_searchsorted(self, device, dtype):
        shapes = [(10,20),(11),(23,19,24),(14,47,12,7),((2,22,4,5,6,7))]
        inputs1 = gendata(shapes, type_list=[dtype], rand_algo="rand")
        inputs2 = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol=1e-5
        for i in range(len(inputs1)):
                input1 = inputs1[i]
                input2 = inputs2[i]
                input_list = [input1, input2]

                runtestapi(func=torch.searchsorted, fwd_input_list=input_list, enable_backward=False, 
                           fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=True)))
    def test_trace(self, device, dtype):
        shapes = [(1,2),(10,20),(20,40),(17,14)]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol=1e-5
        for input in inputs:
            input_list = [input]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.trace, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_tril(self, device, dtype):
        shapes = [(1,2),(10,20,6)]
        diagonals = [2,-1]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol=1e-5

        for input in inputs:
            for diagonal in diagonals:
                input_list = [input, diagonal]
                if dtype in [torch.half,torch.bfloat16]:
                    tol = 1e-3

                runtestapi(func=torch.tril, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_triu(self, device, dtype):
        shapes = [(1,2),(10,20,6)]
        diagonals = [2,-1]
        inputs = gendata(shapes, type_list=[dtype], rand_algo="rand")
        tol=1e-5

        for input in inputs:
            for diagonal in diagonals:
                input_list = [input, diagonal]
                if dtype in [torch.half,torch.bfloat16]:
                    tol = 1e-3
                
                runtestapi(func=torch.triu, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_vander(self, device, dtype):
        Ds = [(3,), (4,), (5,)]
        Ns = [2,3,5]
        increasings = [True, False]
        inputs = gendata(Ds, type_list=[dtype], rand_algo="rand")
        tol=1e-5

        for input in inputs:
            for N in Ns:
                for increasing in increasings:
                    input_list = [input, N, increasing]

                    runtestapi(func=torch.vander, fwd_input_list=input_list, enable_backward=False, 
                               fwd_tol=tol, bwd_tol=tol)


    def test_view_as_real(self):
        shapes = [(1,2),(10,20),(20,40,13),(17,14,11,22)]
        for i in range(len(shapes)):
            a = torch.rand(shapes[i], dtype=torch.cfloat)
            a.requires_grad = True
            out = torch.view_as_real(a)
            out.backward(torch.ones(out.shape))

            a1 = a.detach().clone().cuda()
            a1.requires_grad = True
            out1 = torch.view_as_real(a1)
            out1.backward(torch.ones(out1.shape).cuda())

            assert out1.size() == out.size()
            assert out1.stride() == out.stride()
            assert torch.allclose(out1.cpu(), out)
            assert torch.allclose(a1.grad.cpu(), a.grad)


    def test_view_as_complex(self):
        shapes = [(1,2),(10,20,2),(20,40,13,2),(17,14,11,22,2)]
        for i in range(len(shapes)):
            shape = shapes[i]

            a = torch.rand(shape)
            a.requires_grad = True
            out = torch.view_as_complex(a)
            out.backward(torch.ones(out.shape).to(torch.cfloat))

            a1 = a.detach().clone().cuda()
            a1.requires_grad = True
            out1 = torch.view_as_complex(a1)
            out1.backward(torch.ones(out1.shape).cuda().to(torch.cfloat))

            assert out1.size() == out.size()
            assert out1.stride() == out.stride()
            assert torch.allclose(out1.cpu(), out)
            assert torch.allclose(a1.grad.cpu(), a.grad)

   
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_addbmm(self, device, dtype):

        def test_help(bs, n, m, p):
            tol = 1e-3
            inp1 = make_tensor((n, p), dtype=dtype, device='cpu', low=0, high=1)
            inp2 = make_tensor((bs,n,m), dtype=dtype, device='cpu', low=0, high=1)
            inp3 = make_tensor((bs,m,p), dtype=dtype, device='cpu', low=0, high=1)
            input_list = [inp1, inp2, inp3]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            runtestapi(func=torch.addbmm, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        bss = [3,4,5]
        ns = [3,4,5]
        ms = [8,7,6]
        ps = [12,24]
        for bs in bss:
            for n in ns:
                for m in ms:
                    for p in ps:
                        test_help(bs,n,m,p)
                        

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_addmm(self, device, dtype):

        def test_help(n,m,p):
            tol = 1e-3
            inp1 = make_tensor((n,p), dtype=dtype, device='cpu', low=0, high=1)
            inp2 = make_tensor((n,m), dtype=dtype, device='cpu', low=0, high=1)
            inp3 = make_tensor((m,p), dtype=dtype, device='cpu', low=0, high=1)
            input_list = [inp1, inp2, inp3]
            if dtype in [torch.half,torch.bfloat16]:
                tol = 1e-2
            
            runtestapi(func=torch.addmm, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        ns = [3,4,5]
        ms = [8,7,6]
        ps = [12,24]
        for n in ns:
            for m in ms:
                for p in ps:
                    test_help(n,m,p)
                    

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_addmv(self, device, dtype):
        def test_help(n,m):
            tol = 1e-4
            inp1 = make_tensor((n,), dtype=dtype, device='cpu', low=0, high=1)
            inp2 = make_tensor((n,m), dtype=dtype, device='cpu', low=0, high=1)
            inp3 = make_tensor((m,), dtype=dtype, device='cpu', low=0, high=1)
            input_list = [inp1, inp2, inp3]
            if dtype in [torch.half,torch.bfloat16]:  
                tol=1e-2

            runtestapi(func=torch.addmv, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        ns = [3,4,5]
        ms = [8,9,10]
        for n in ns:
            for m in ms:
                test_help(n,m)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_addr(self, device, dtype):

        def test_help(n,m):
            tol = 1e-4
            inp1 = make_tensor((n,m), dtype=dtype, device='cpu', low=0, high=1)
            inp2 = make_tensor((n,), dtype=dtype, device='cpu', low=0, high=1)
            inp3 = make_tensor((m,), dtype=dtype, device='cpu', low=0, high=1)
            input_list = [inp1, inp2, inp3]
            if dtype in [torch.half,torch.bfloat16]: 
                tol = 1e-2
            
            runtestapi(func=torch.addr, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        ns = [3,4,5]
        ms = [8,7,6]
        for n in ns:
            for m in ms:
                test_help(n,m)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_baddbmm(self, device, dtype):

        def test_help(bs,n,m,p):
            tol = 1e-3
            inp1 = make_tensor((bs,n,p), dtype=dtype, device='cpu',low=0, high=1)
            inp2 = make_tensor((bs,n,m), dtype=dtype, device='cpu',low=0, high=1)
            inp3 = make_tensor((bs,m,p), dtype=dtype, device='cpu',low=0, high=1)
            input_list = [inp1, inp2, inp3]
            if dtype in [torch.half,torch.bfloat16]:  
                tol=1e-2
            
            runtestapi(func=torch.baddbmm, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        bss = [2,4]
        ns = [3,4]
        ms = [8,7]
        ps = [2,4]
        for bs in bss:
            for n in ns:
                for m in ms:
                    for p in ps:
                        test_help(bs,n,m,p)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False)))
    def test_logadd(self, device, dtype):
        input_shape_list = [(5)]
        other_shape_list = [(5)]

        fn_list = [{"fn": torch.logaddexp, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}, "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}, 
                    {"fn": torch.logaddexp2, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}, "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        inputs = gendata(input_shape_list, type_list=[dtype])   # [(info, data)]
        others = gendata(other_shape_list, type_list=[dtype])
        tol = 1e-4
        for input in inputs:    
            for other in others:
                input_list = [input, other]
                if dtype in [torch.half,torch.bfloat16]: 
                    tol = 1e-2

                    runtestapi(func=torch.logaddexp, fwd_input_list=input_list, enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol)

    
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_bmm(self, device, dtype):
        def test_help(bs,n,m,p):
            tol = 1e-3
            inp1 = make_tensor((bs,n,m), dtype=dtype, device='cpu', low=0, high=1)
            inp2 = make_tensor((bs,m,p), dtype=dtype, device='cpu', low=0, high=1)
            input_list = [inp1, inp2]
            if dtype in [torch.half,torch.bfloat16]: 
                tol = 1e-2
            
            runtestapi(func=torch.bmm, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        bss = [2,4]
        ns = [3,4,5]
        ms = [8,7,6]
        ps = [2,4]
        for bs in bss:
            for n in ns:
                for m in ms:
                    for p in ps:
                        test_help(bs,n,m,p)


    @onlyCUDA   
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_chain_matmul(self,device,dtype):
        rol = 1e-2
        a = make_tensor((3,4), dtype=dtype, device='cpu', requires_grad=True, low=0, high=1)
        b = make_tensor((4,5), dtype=dtype, device='cpu', requires_grad=True, low=0, high=1)
        c = make_tensor((5,6), dtype=dtype, device='cpu', requires_grad=True, low=0, high=1)
        d = make_tensor((6,7), dtype=dtype, device='cpu', requires_grad=True, low=0, high=1)
        e = make_tensor((7,8), dtype=dtype, device='cpu', requires_grad=True, low=0, high=1)
        out = torch.chain_matmul(a,b,c,d,e)
        out.backward(torch.ones(out.shape).to(dtype=dtype))

        a1 = a.detach().clone().cuda()
        b1 = b.detach().clone().cuda()
        c1 = c.detach().clone().cuda()
        d1 = d.detach().clone().cuda()
        e1 = e.detach().clone().cuda()
        a1.requires_grad = True
        b1.requires_grad = True
        c1.requires_grad = True
        d1.requires_grad = True
        e1.requires_grad = True
        out1 = torch.chain_matmul(a1,b1,c1,d1,e1)
        out1.backward(torch.ones(out1.shape).cuda().to(dtype=dtype))

        assert torch.allclose(out1.cpu(), out, rtol=rol, atol=rol)
        assert torch.allclose(a1.grad.cpu(), a.grad, rtol=rol, atol=rol)
        assert torch.allclose(b1.grad.cpu(), b.grad, rtol=rol, atol=rol)
        assert torch.allclose(c1.grad.cpu(), c.grad, rtol=rol, atol=rol)
        assert torch.allclose(d1.grad.cpu(), d.grad, rtol=rol, atol=rol)
        assert torch.allclose(e1.grad.cpu(), e.grad, rtol=rol, atol=rol)
    
    
    @onlyCUDA   
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_cholesky(self,device,dtype):
        shapes = [3,4,5,6,7,8,9]
        rol = 1e-3
        for n in shapes:
            a = torch.rand((n,n),device=device,dtype=dtype)
            inp = torch.mm(a, a.t())
            out = torch.cholesky(inp)

            out_ori = torch.mm(out, out.t())
            assert  torch.allclose(out_ori.cpu(), inp.cpu(), rtol=rol, atol=rol)


    @onlyCUDA   
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_cholesky_reverse(self,device,dtype):
        shapes = [3,4,5,6,7,8,9]
        rol = 1e-3
        for n in shapes:
            a = torch.rand((n,n),device=device,dtype=dtype)
            a = torch.mm(a, a.t()) + (1e-5*torch.eye(n)).cuda()
            u = torch.cholesky(a)
            out = torch.cholesky_inverse(u)

            assert  torch.allclose(out.cpu(), a.inverse().cpu(), rtol=rol, atol=rol)



instantiate_device_type_tests(TestTorchFunctionSegment5, globals())

if __name__ == "__main__":
    unittest.main()

