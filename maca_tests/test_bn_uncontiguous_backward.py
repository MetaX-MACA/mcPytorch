import torch
import copy


for dtype in [torch.float16, torch.float32, ]:
  rtol = 1e-3 if dtype == torch.float16 else 1e-5
  atol = 1e-3 if dtype == torch.float16 else 1e-5

  for shape in [(32, 2048, 25, 42), (32, 64, 400, 672), (32, 64, 200, 336), (32, 256, 200, 336), (32, 128, 200, 336),
                (32, 128, 100, 168), (32, 512, 100, 168), (32, 256, 100, 168), (32, 1024, 50, 84), (32, 256, 50, 84),
                (32, 512, 50, 84), (32, 512, 25, 42), (32, 64, 400, 640), (32, 64, 200, 320), (32, 256, 200, 320), 
                (32, 128, 200, 320), (32, 128, 100, 160), (32, 512, 100, 160), (32, 256, 50, 80), (32, 256, 100, 160),
                (32, 1024, 50, 80), (32, 2048, 25, 40), (32, 512, 25, 40), (32, 256, 312, 200),(32, 64, 312, 200)]:
      m = torch.nn.BatchNorm2d(shape[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      m_d = copy.deepcopy(m).to("cuda").to(dtype)
      m_d.eval()
      m.eval()

      input = torch.randn(shape, dtype = torch.float32, device="cpu").requires_grad_(True)
      input_d = input.clone().detach().cuda().to(dtype).requires_grad_(True)
      g_input = torch.randn(shape)
      g_input_d = g_input.clone().detach().cuda().to(dtype)
      output_d = m_d(input_d)
      output = m(input)

      output_d.backward(g_input_d)
      output.backward(g_input)

      input_grad = input.grad
      input_d_grad = input_d.grad

      print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(input_d_grad.cpu().float() - input_grad)))
      
      if not torch.allclose(input_d_grad.cpu().float(), input_grad, rtol=rtol, atol=atol):
        print("##### fail")
        exit(1)
print("##### pass")
exit(0)
