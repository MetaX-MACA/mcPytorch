import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--type", default="checkin", help="<checkin|daily>")
args = parser.parse_args()

def test_accuracy(op, test_cases):
    torch.manual_seed(0)
    for id, test_case in test_cases.items():
        gpu_device = torch.device('cuda')
        dtype = test_case["dtype"] if "dtype" in test_case else torch.float32
        x = torch.randn(test_case["input_shape"], dtype=dtype) if dtype != torch.int64 else (torch.randn(test_case["input_shape"], dtype=torch.float) * 100).to(dtype)
        y_gpu = op(x.to(gpu_device), test_case["dim"])

        cpu_device = torch.device('cpu')
        y_cpu = op(x.to(cpu_device), test_case["dim"])
        rtol = 1e-1
        atol = 1
        if "rtol" in test_case:
            rtol = test_case["rtol"]
        if "atol" in test_case:
            atol = test_case["atol"]
        all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=rtol, atol=atol)
        print("{}/{}: all close: {}".format(id, len(test_cases), all_close))
        if not all_close:
            print("input_shape: {}, reduce dim: {}".format(test_case["input_shape"], test_case["dim"]))
            print("gpu: {}".format(y_gpu))
            print("cpu: {}".format(y_cpu))
            exit(1)

def test_max_accuracy(test_cases):
    torch.manual_seed(0)
    for id, test_case in test_cases.items():
        gpu_device = torch.device('cuda')
        dtype = test_case["dtype"] if "dtype" in test_case else torch.float32
        x = torch.randn(test_case["input_shape"], dtype=dtype) if dtype != torch.int64 else (torch.randn(test_case["input_shape"], dtype=torch.float) * 100).to(dtype)
        y_gpu, index_gpu = torch.max(x.to(gpu_device), test_case["dim"])

        cpu_device = torch.device('cpu')
        y_cpu, index_cpu = torch.max(x.to(cpu_device), test_case["dim"])
        all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
        all_close_1 = torch.allclose(index_gpu.cpu(), index_cpu, rtol=1e-4, atol=1e-7)
        print("{}: all close: {}".format(id, all_close))
        print("{}: idx all close: {}".format(id, all_close_1))
        if not all_close or not all_close_1:
            print("input_shape: {}".format(test_case["input_shape"]))
            print("input: {}".format(x))
            print("dtype: {}".format(dtype))
            print("gpu: {}".format(y_gpu))
            print("cpu: {}".format(y_cpu))
            print("gpu idx: {}".format(index_gpu))
            print("cpu idx: {}".format(index_cpu))
            exit(1)


def test_sum_accuracy():
    sum_test_cases = {
      0: {"input_shape": [105644032], "dim": None},
      1: {"input_shape": [11272192], "dim": None},
      2: {"input_shape": [119], "dim": None},
      3: {"input_shape": [12582912], "dim": None},
      4: {"input_shape": [26], "dim": None},
      5: {"input_shape": [3072], "dim": None},
      6: {"input_shape": [4096], "dim": None},
      7: {"input_shape": [4194304], "dim": None},
      8: {"input_shape": [2, 721, 1440], "dim": [-2, -1]}
    }
    print("======test sum accuracy===========")
    test_accuracy(torch.sum, sum_test_cases)



def test_argmax_accuracy():
    argmax_test_cases = {
      0: {"input_shape": 105644032, "dim": None},
      1: {"input_shape": 11272192, "dim": None},
      2: {"input_shape": 119, "dim": None},
      3: {"input_shape": 12582912, "dim": None},
      4: {"input_shape": 26, "dim": None},
      5: {"input_shape": 3072, "dim": None},
      6: {"input_shape": 4096, "dim": None},
      7: {"input_shape": 4194304, "dim": None}
    }
    print("======test argmax accuracy===========")
    test_accuracy(torch.argmax, argmax_test_cases)

def test_mean_accuracy():
    mean_test_cases = {
      0: {"input_shape": 1, "dim": None},
      1: {"input_shape": 4, "dim": None},
      2: {"input_shape": 119, "dim": None},
      3: {"input_shape": [1, 1, 1, 2, 768, 768], "dim": [0, 1, 2, 4, 5]},
    }
    print("======test mean accuracy===========")
    test_accuracy(torch.mean, mean_test_cases)

def test_official_reduce():
    # China mobile
    prod_test_cases = {
      0: {"input_shape": [1, 2], "dim": 1, "dtype": torch.float},
    }
    test_accuracy(torch.prod, prod_test_cases)
    # # InternLLM
    max_test_cases = {
      0: {"input_shape": [2048, 103168], "dim": 1, "dtype": torch.bfloat16},
    }
    test_max_accuracy(max_test_cases)
    argmax_test_cases = {
      0: {"input_shape": [2048, 103168], "dim": 1, "dtype": torch.bfloat16},
      1: {"input_shape": [2048, 32000], "dim": 1, "dtype": torch.bfloat16},
    }
    test_accuracy(torch.argmax, argmax_test_cases)
    sum_test_cases = {
      0: {"input_shape": [2048, 103168], "dim": 1, "dtype": torch.bfloat16},
      0: {"input_shape": [2048, 32000], "dim": 1, "dtype": torch.bfloat16},
    }
    test_accuracy(torch.sum, sum_test_cases)

    # Transformer
    std_var_cases = {
      0: {"input_shape": [16384, 512], "dim": 1, "dtype": torch.float}
    }
    test_accuracy(torch.std, std_var_cases)

    max_test_cases = {
      0: {"input_shape": [4000, 4000], "dim": 0, "dtype": torch.float}
    }
    test_max_accuracy(max_test_cases)
    # spagcn
    sum_test_cases = {
      0: {"input_shape": [49921, 9, 50], "dim": 1, "dtype": torch.float},
    }
    test_accuracy(torch.sum, sum_test_cases)

    # bonito
    max_test_cases = {
      0: {"input_shape": [184320, 5120], "dim": 1, "dtype": torch.float},
    }
    test_max_accuracy(max_test_cases)

    # ChateGLM
    sum_test_cases = {
      0: {"input_shape": [8, 16, 2048, 128], "dim": 1, "dtype": torch.float16},
    }
    test_accuracy(torch.sum, sum_test_cases)

    # Mask-RCNN
    sum_test_cases = {
      0: {"input_shape": [2, 256, 115200], "dim": [0, 2], "dtype": torch.float16},
    }
    test_accuracy(torch.sum, sum_test_cases)

    # Segformer
    mean_test_cases = {
      0: {"input_shape": [1, 67108864], "dim": 0, "dtype": torch.float},
      1: {"input_shape": [67108864, 1], "dim": 1, "dtype": torch.float},
    }
    test_accuracy(torch.mean, mean_test_cases)

    # MMLab-mmpre
    max_test_cases = {
      0: {"input_shape": [1, 1000], "dim": 0, "dtype": torch.int64},
      1: {"input_shape": [67108864, 9], "dim": 1, "dtype": torch.half},
    }
    test_max_accuracy(max_test_cases)


def test_corner_reduce():
    reduce_test_cases = {}
    index = 0
    for cols in [4, 8, 16, 32, 64, 65, 128, 512, 1023, 2048, 3000, 4041, 5128, 9888]:
        for rows in [3, 4, 5, 16, 33, 48, 64, 129, 256, 512, 1025, 2000, 3132, 4538, 6278, 9238]:
            for dim in [0, 1]:
                for dtype in [torch.float, torch.bfloat16]:
                    reduce_test_cases[index] = {"input_shape": [cols, rows], "dim": dim, "dtype": dtype}
                    index += 1
    test_max_accuracy(reduce_test_cases)
    test_accuracy(torch.sum, reduce_test_cases)
    test_accuracy(torch.amin, reduce_test_cases)

def test_3d_corner_reduce():
    reduce_test_cases = {}
    index = 0
    if args.type == "checkin":
        dim0_list = [32, 2048, 53324]
        dim1_list = [64, 3132, 65532]
        dim2_list = [75, 1135, 35235]
    else:
        dim0_list = [4, 32, 65, 2048, 9888, 53324]
        dim1_list = [3,  64, 512, 1025, 3132, 65532]
        dim2_list = [12,  75, 1135, 5236, 35235, 65532]

    for dim0 in dim0_list:
        for dim1 in dim1_list:
          for dim2 in dim2_list:
              if dim0 * dim1 * dim2 > 65536 * 65536:
                  continue
              for reduce_dim in [1]:
                  for dtype in [torch.float, torch.bfloat16]:
                      reduce_test_cases[index] = {"input_shape": [dim0, dim1, dim2], "dim": reduce_dim, "dtype": dtype, "rtol": 1e-2, "atol": 10}
                      index += 1
    torch.cuda.empty_cache()
    test_accuracy(torch.sum, reduce_test_cases)
    torch.cuda.empty_cache()

def test_slice_reduce():
    x = torch.randn((5,5,5), device="cuda", dtype=torch.float32)[::2, ::3, 1:-1:2]
    y_gpu = torch.max(x)
    cpu_device = torch.device('cpu')
    y_cpu = torch.max(x.to(cpu_device))
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

def test_as_strided_reduce():
    x = torch.randn((5,5,5), device="cuda", dtype=torch.bfloat16).as_strided((5,3,3), (25,10,2))
    y_gpu = torch.sum(x)
    cpu_device = torch.device('cpu')
    y_cpu = torch.sum(x.to(cpu_device))
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    x = torch.randn((15, 10), device="cuda", dtype=torch.bfloat16).as_strided((7, 5), (20,2))
    y_gpu, _ = torch.max(x, 1)
    y_cpu, _ = torch.max(x.to(cpu_device), 1)
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    x = torch.randn((15, 10), device="cuda", dtype=torch.bfloat16).as_strided((7, 10), (20,1))
    y_gpu, _ = torch.max(x, 0)
    y_cpu, _ = torch.max(x.to(cpu_device), 0)
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    x = torch.randn(4352, 24, 128, dtype=torch.float16, device="cuda").as_strided((24, 4352, 128), (128, 3072, 1))
    y_gpu = torch.mean(x, 2)
    y_cpu = torch.mean(x.to(cpu_device), 2)
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-4)
    if not all_close:
        exit(1)

    x = torch.randn((128, 900, 16, 72), device="cuda", dtype=torch.float32).as_strided((128, 16, 900, 72), (103680, 72, 1152, 1))
    y_gpu = torch.sum(x, 3)
    cpu_device = torch.device('cpu')
    y_cpu = torch.sum(x.to(cpu_device), 3)
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-4)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    x = torch.randn((24, 10), device="cuda", dtype=torch.bfloat16).as_strided((12, 10), (20,1))
    y_gpu, _ = torch.max(x, 0)
    y_cpu, _ = torch.max(x.to(cpu_device), 0)
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    y_gpu = torch.sum(x, 0)
    y_cpu = torch.sum(x.to(cpu_device), 0)
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    x = torch.randn(1000000, 6, 8, dtype=torch.float32, device="cuda").as_strided((8, 6, 8), (196, 13, 1))
    y_gpu = torch.sum(x, 1)
    y_cpu = torch.sum(x.to(cpu_device), 1)
    all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)


def test_transpose_reduce():
  x_base = torch.randn(1000000000, device="cuda:0")
  size = 3
  for length in [250, 500, 750]:
    x = x_base.as_strided((1, 256, length, length, size), (256*size, size, length * 256*size, 256*size, 1))
    x_cpu = x.cpu()
    res = torch.linalg.vector_norm(x, dim=-1)
    res_cpu = torch.linalg.vector_norm(x_cpu, dim=-1)
    assert (torch.allclose(res.cpu(), res_cpu, rtol=1e-4, atol=1e-3)), f"InputPerOutputContinuousReduceKernelTranspose result wrong"

def test_as_over_int32_size_reduce():
    int32_size = 2147483647
    max_test_cases = {
      0: {"input_shape": [10, int32_size+3], "dim": 1, "dtype": torch.bfloat16},
      1: {"input_shape": [11, int32_size*2+2], "dim": 1, "dtype": torch.bfloat16},
      2: {"input_shape": [100, int32_size*8+1], "dim": 1, "dtype": torch.bfloat16},
    }
    test_max_accuracy(max_test_cases)

    sum_test_cases = {
      0: {"input_shape": [int32_size+3], "dim": None, "dtype": torch.bfloat16},
      1: {"input_shape": [int32_size*2+2], "dim": None, "dtype": torch.bfloat16},
      2: {"input_shape": [int32_size*8+1], "dim": None, "dtype": torch.bfloat16},
    }
    test_accuracy(torch.sum, sum_test_cases)

def test_output_non_continuguous():
    a_cpu = torch.randn(5, 4)
    a_gpu = a_cpu.cuda()
    b_cpu = torch.zeros(5, 5)
    b_gpu = b_cpu.cuda()
    # all to one
    c_cpu = b_cpu[:, 1]
    c_gpu = b_gpu[:, 1]
    torch.logsumexp(a_cpu, dim=(0,1), keepdim=False, out=c_cpu)
    torch.logsumexp(a_gpu, dim=(0,1), keepdim=False, out=c_gpu)
    all_close = torch.allclose(c_gpu.cpu(), c_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    # continguous
    torch.logsumexp(a_cpu, 1, out=c_cpu)
    torch.logsumexp(a_gpu, 1, out=c_gpu)
    all_close = torch.allclose(c_gpu.cpu(), c_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

    # imcontinguous
    torch.logsumexp(a_cpu, 0, out=c_cpu)
    torch.logsumexp(a_gpu, 0, out=c_gpu)
    all_close = torch.allclose(c_gpu.cpu(), c_cpu, rtol=1e-4, atol=1e-7)
    print("all close: {}".format(all_close))
    if not all_close:
        exit(1)

def test_extreme_reduce():
    reduce_test_cases = {}
    index = 0
    for cols in [1, 6710]:
        for rows in [1, 6710]:
            for dim in [0, 1]:
                for dtype in [torch.float, torch.bfloat16]:
                  reduce_test_cases[index] = {"input_shape": [cols, rows], "dim": dim, "dtype": dtype}
                  index += 1
    test_max_accuracy(reduce_test_cases)
    test_accuracy(torch.sum, reduce_test_cases)
    test_accuracy(torch.mean, reduce_test_cases)

    sum_test_cases = {
      0: {"input_shape": [1, 67108864], "dim": 0, "dtype": torch.float},
      1: {"input_shape": [67108864, 1], "dim": 1, "dtype": torch.float},
    }
    test_accuracy(torch.sum, sum_test_cases)

def test_bool_view_as_uint8():
  t = torch.tensor([[False,  True, False, False], [ True, False,  True,  True], [False,  True, False,  True]], device='cuda')

  def convert_boolean_tensors(x):

      # Map False -> 0 and True -> Random value in [2, 255]
      true_vals = torch.randint(2, 255, x.shape, dtype=torch.uint8, device=x.device)
      false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
      x_int = torch.where(x, true_vals, false_vals)
      ret = x_int.view(torch.bool)
      return ret

  #torch.manual_seed(0)
  transformed_t = convert_boolean_tensors(t)
  y = torch.sum(transformed_t, 0, dtype=torch.bool)
  y_bool = torch.sum(t, 0, dtype=torch.bool)
  all_close = torch.allclose(y, y_bool)
  print(y)
  print(y_bool)
  if not all_close:
      print("test_bool_view_as_uint8 failed")
      exit(1)

def test_daily_build_case():
  argmax_test_cases = {
      0: {"input_shape": [137500000], "dim": None, "dtype": torch.float64},
    }
  test_accuracy(torch.argmax, argmax_test_cases)
  test_accuracy(torch.argmin, argmax_test_cases)




def test_count_nonzero():
  dtypes = [torch.int8, torch.int32, torch.int64]
  torch.manual_seed(1)
  col = 225
  for dtype in dtypes:
      inp = torch.randint(low=-30, high=30, size=(225,), dtype=dtype,device='cuda')
      inp_cpu = inp.cpu()

      res_cpu = torch.count_nonzero(inp_cpu)
      res_gpu = torch.count_nonzero(inp)
      assert torch.allclose(res_cpu, res_gpu)

def test_checkin_case():
    test_count_nonzero()

if __name__ == "__main__":
    if args.type == "checkin":
      test_checkin_case()
      test_3d_corner_reduce()
    if args.type == "daily":
      test_daily_build_case()
      test_official_reduce()
      test_corner_reduce()
      test_slice_reduce()
      test_as_strided_reduce()
      test_transpose_reduce()
      test_output_non_continuguous()
      test_sum_accuracy()
      test_argmax_accuracy()
      test_mean_accuracy()
      test_3d_corner_reduce()
      test_extreme_reduce()
      test_bool_view_as_uint8()
