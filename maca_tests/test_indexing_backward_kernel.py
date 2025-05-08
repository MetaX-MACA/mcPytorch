import torch
import argparse

torch.manual_seed(0)

def test_index_put(shape, dtype=torch.bfloat16):
    input = torch.randn(shape, device="cuda", dtype=dtype)
    index_max = min(shape)
    index = torch.randint(0, index_max, (1, shape[0]), device="cuda")
    value = torch.randn(1, shape[0], shape[1], device="cuda", dtype=dtype)

    output = input.index_put([index], value, accumulate=True)

    input_c = input.cpu()
    index_c = index.cpu()
    value_c = value.cpu()
    output_c = input_c.index_put([index_c], value_c, accumulate=True)

    if not torch.allclose(output_c, output.cpu()):
        print(f"test_index_put error with shape:{shape}!")
        exit(1)

def test_index_put_uncontiguous(shape, dtype=torch.bfloat16):
      input = torch.randn(shape, device="cuda", dtype=dtype)
      index_max = min(shape)
      index = torch.randint(0, index_max, (1, shape[0]), device="cuda")
      value = torch.randn(1, shape[0], shape[1], device="cuda", dtype=dtype)
      value = value.as_strided((1, shape[0], shape[1]), (shape[0], 1, 2))

      output = input.index_put([index], value, accumulate=True)

      input_c = input.cpu()
      index_c = index.cpu()
      value_c = value.cpu()
      output_c = input_c.index_put([index_c], value_c, accumulate=True)

      if not torch.allclose(output_c, output.cpu()):
         print(f"test_index_put_uncontiguous error with shape:{shape}!")
         exit(1)

def test_vec_index_put():
   input_shape_list = [(33,96), (133,196), (200,300), (235,256), (444,512)]
   value_shape_list = [(3329,96), (6000,196), (5992,300), (4444,256), (5555,512)]

   dtype_list=[torch.half,torch.float,torch.bfloat16]

   for i in range(len(input_shape_list)):
      input_shape = input_shape_list[i]
      value_shape = value_shape_list[i]
      for dtype in dtype_list:
         input = torch.zeros(input_shape,device="cuda",dtype=dtype)
         value = torch.rand(value_shape,device="cuda",dtype=dtype)
         index = [torch.randint(0,input_shape[0],(value_shape[0],),device="cuda")]
         out =  input.index_put(index,value,accumulate=True)

         inputc = input.cpu()
         valuec = value.cpu()
         indexc = [index[0].cpu()]
         outc =  inputc.index_put(indexc,valuec,accumulate=True)

         res = torch.allclose(out.cpu(),outc)
         if not res:
               exit(1)

test_vec_index_put()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  if args.type == "checkin":
     for s0 in [3, 8, 2048, 2047]:
        for s1 in range(1, 13):
           shape = [s0, s1*64]
           test_index_put(shape)
           
  else:
     for s0 in range(2, 5120, 51):
        for s1 in range(1, 20):
           shape = [s0, s1*64]
           test_index_put(shape)
           test_index_put_uncontiguous(shape)