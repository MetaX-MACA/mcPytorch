import torch
import torch.nn as nn
import copy

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

for dtype in [torch.float16, torch.float32, torch.bfloat16]:
    for shape in [(4096, 8000), (4095, 8000), (4095, 7999), (64, 64), (63, 65), (65, 63)]:
        for reduction in ["mean"]:
            for ignore_index in [0,2,10]:
                loss_fn = nn.NLLLoss(reduction=reduction,ignore_index=ignore_index)
                input = torch.randn(shape, dtype = dtype, device="cpu")
                input.requires_grad = True
                target = torch.randint(0, shape[1], (shape[0],))
                input_d = input.detach().cuda()
                input_d.requires_grad = True
                target_d = target.cuda()
                output_d = loss_fn(input_d, target_d)
                output_d.backward()
                if dtype == torch.float16 or dtype == torch.bfloat16:
                  output_golden = loss_fn(input.float(), target)
                  output_golden.backward()
                  print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output_d.cpu().float() - output_golden)))
                  if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
                    exit(1)
                  print("grad: type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(input_d.grad.cpu().float() - input.grad.float())))
                  if not torch.allclose(input_d.grad.cpu().float(), input.grad.float(), rtol=5e-4, atol=5e-4):
                    exit(1)
                else:
                  output_golden = loss_fn(input, target)
                  output_golden.backward()
                  print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output_d.cpu() - output_golden)))
                  if not torch.allclose(output_d.cpu().float(), output_golden):
                    exit(1)
                  print("grad: type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(input_d.grad.cpu() - input.grad)))
                  if not torch.allclose(input.grad.cpu().float(), input.grad):
                    exit(1)
exit(0)

