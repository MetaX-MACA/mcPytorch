import torch
import torch.nn.functional as F
import argparse

def launch(shape):
    for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        for is_log_softmax in [True, False]:
            input = torch.randn(shape, dtype = dtype, device="cpu")
            input_d = input.cuda()
            if is_log_softmax:
                output_d = F.log_softmax(input_d, dim=-1)
            else:
                output_d = F.softmax(input_d, dim=-1)

            if dtype == torch.float16 or dtype == torch.bfloat16:
                if is_log_softmax:
                    output_golden = F.log_softmax(input.float(), dim=-1)
                    print("logsoftmax: type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output_d.cpu().float() - output_golden)))
                    if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-3, atol=5e-3):
                        exit(1)
                else:
                    output_golden = F.softmax(input.float(), dim=-1)
                    print("softmax: type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output_d.cpu().float() - output_golden)))
                    if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
                        exit(1)
            else:
                if is_log_softmax:
                    output_golden = F.log_softmax(input, dim=-1)
                    print("logsoftmax: type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output_d.cpu() - output_golden)))
                else:
                    output_golden = F.softmax(input, dim=-1)
                    print("softmax: type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output_d.cpu() - output_golden)))
                if not torch.allclose(output_d.cpu().float(), output_golden):
                    exit(1)
def test(t):
    if type == "checkin" or type == "daily":
        for shape in [(2, 5, 4096, 4096), (4096, 8000)]:
            launch(shape)
    if type == "daily":
        for shape in [(2, 32, 2048, 2048), (16, 6400, 6400), (16, 2304, 2304), (16, 9216, 9216)]:
            launch(shape)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()
  test(args.type)
