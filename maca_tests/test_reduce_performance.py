import torch


def profiler(x, op, dim):
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    with torch.no_grad():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,],
            schedule=torch.profiler.schedule(
            wait=0,
            warmup=5,
            active=20,
            repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True) as p:
            for i in range(26):
                z = op(x,dim)
                p.step()


def test_sum_performance():
    sum_test_cases = {
      0: {"input_shape": [105644032], "dim": None},
      1: {"input_shape": [11272192], "dim": None},
      2: {"input_shape": [119], "dim": None},
      3: {"input_shape": [12582912], "dim": None},
      4: {"input_shape": [26], "dim": None},
      5: {"input_shape": [3072], "dim": None},
      6: {"input_shape": [4096], "dim": None},
      7: {"input_shape": [4194304], "dim": None},
      # Intern LLM
      8: {"input_shape": [2048, 103168], "dim": 1},
      9: {"input_shape": [2048, 32000], "dim": 1}
    }
    torch.manual_seed(0)
    device = torch.device('cuda')
    for _, test_case in sum_test_cases.items():
        print("input_shape: {}".format(test_case["input_shape"]))
        x = torch.randn(test_case["input_shape"], dtype=torch.bfloat16).to(device)
        profiler(x, torch.sum, test_case["dim"])

def test_argmax_performance():
    # Intern LLM
    argmax_test_cases = {
      0: {"input_shape": [2048, 103168], "dim": 1},
      1: {"input_shape": [2048, 32000], "dim": 1},
    }
    torch.manual_seed(0)
    device = torch.device('cuda')
    for _, test_case in argmax_test_cases.items():
        print("input_shape: {}".format(test_case["input_shape"]))
        x = torch.randn(test_case["input_shape"], dtype=torch.bfloat16).to(device)
        profiler(x, torch.argmax, test_case["dim"])

if __name__ == "__main__":
    test_sum_performance()
    test_argmax_performance()
    