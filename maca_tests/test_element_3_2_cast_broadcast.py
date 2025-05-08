import torch
import copy

def test_element_3_2_cast_broadcast(shapes, dtype1, dtype2):
  for shape in shapes:
    a = torch.randn(shape, dtype=dtype1)
    a = a.transpose(0, 2)

    b = torch.randn(shape, dtype=dtype2)
    b = b.transpose(0, 2)

    b_c = copy.deepcopy(b)
    b_c = b_c.as_strided(b_c.shape, (1, 0, b_c.shape[0]))
    ref = a + b_c

    a_d = a.cuda()
    b_d = b.cuda()
    b_d = b_d.as_strided(b_d.shape, (1, 0, b_d.shape[0]))
    out = a_d + b_d
    if not torch.allclose(ref, out.cpu()):
      return False
  return True

def test_element_3_2_cast_broadcast_uncontiguous(shapes, dtype1, dtype2):
  for shape in shapes:
    shape_broad = (shape[0], 1, shape[2])
    a = torch.randn(shape[0]*shape[1]*shape[2]*3, dtype=dtype1, device="cuda")
    b = a.as_strided(shape, (3*shape[1]*shape[2],3*shape[2],1))
    c = torch.randn(shape_broad, dtype=dtype2, device="cuda")
    c = c.as_strided(shape_broad, (shape[2], 0, 1))

    d = b + c
    b_cpu = b.cpu()
    c_cpu = c.cpu()
    d_cpu = b_cpu + c_cpu
    res=torch.allclose(d.cpu(), d_cpu)
    if not res:
        return False
  return True

if __name__ == "__main__":
  # out: float32, shape=[s0,s1,s2], stride=[1,s0,s0*s1]
  # arg0: half, shape=[s0,s1,s2], stride=[1,s0,s0*s1]
  # arg1: float32 (broadcast), shape=[s0,s1,s2], stride=[1,0,s0]
  # the shape include in bert
  shapes = [[24, 4608, 384], [32, 4608, 384], [64, 4608, 384], [128, 4608, 384], [21, 4608, 384]]
  # different vec num
  vec_shapes = [[24, 4096, 64], [24, 4096, 128], [24, 4096, 192], [24, 4096, 256], [24, 4096, 320]]
  # the shape trigger y_remain, x_reamin
  ext_shapes = [[24, 4096, 132], [24, 4096, 133], [23, 4099, 133]]

  shapes.extend(vec_shapes)
  shapes.extend(ext_shapes)

  if not test_element_3_2_cast_broadcast(shapes, torch.float16, torch.float32):
    exit(1)

  # uncontiguous shapes
  # arg0: half, shape=[s0,s1,s2], stride=[1,3*s0,3*s0*s1]
  un_shapes = [[128, 784, 384]]
  if not test_element_3_2_cast_broadcast_uncontiguous(shapes, torch.float16, torch.float32):
    exit(1)

  exit(0)
