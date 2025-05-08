import torch
import copy
import argparse

def test_elementwise_3_2_broadcast_dim1_contiguous(shapes, dtype):
  # out: dtype, shape=[s0,s1,s2], stride=[1,s0,s0*s1]
  # arg0: dtype, shape=[s0,s1,s2], stride=[1,s0,s0*s1]
  # arg1: dtype (broadcast), shape=[s0,s1,s2], stride=[1,0,!=0]
  for shape in shapes:
    a = torch.randn(shape, dtype=dtype)
    a = a.transpose(0, 2)

    b = torch.randn(shape, dtype=dtype)
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

def test_elementwise_3_2_broadcast_dim1_uncontiguous(shapes, dtype):
  for shape in shapes:
    a = torch.randn(shape[0] * shape[1], shape[1], shape[2] * 4, dtype=dtype)
    a = a.transpose(0, 2)

    b = torch.randn(shape, dtype=dtype)
    b = b.transpose(0, 2)

    a_c = copy.deepcopy(a)
    b_c = copy.deepcopy(b)
    a_c = a_c.as_strided(b.shape, (2, b.shape[0] * 4, b.shape[2] // 2))
    b_c = b_c.as_strided(b.shape, (2, 0, b.shape[0] * 2))
    ref = a_c * b_c

    a_c_2 = copy.deepcopy(a)
    b_c_2 = copy.deepcopy(b)
    a_c_2 = a_c_2.as_strided(b.shape, (2, b.shape[0], b.shape[2] // 2))
    b_c_2 = b_c_2.as_strided(b.shape, (2, 0, b.shape[0] * 2))
    ref_2 = a_c_2 * b_c_2

    a_d = copy.deepcopy(a).cuda()
    b_d = copy.deepcopy(b).cuda()
    a_d = a_d.as_strided(b.shape, (2, b.shape[0] * 4, b.shape[2] // 2))
    b_d = b_d.as_strided(b.shape, (2, 0, b.shape[0] * 2))
    out = a_d * b_d

    a_d_2 = copy.deepcopy(a).cuda()
    b_d_2 = copy.deepcopy(b).cuda()
    a_d_2 = a_d_2.as_strided(b.shape, (2, b.shape[0], b.shape[2] // 2))
    b_d_2 = b_d_2.as_strided(b.shape, (2, 0, b.shape[0] * 2))
    out_2 = a_d_2 * b_d_2

    if not torch.allclose(ref, out.cpu()) or not torch.allclose(ref_2, out_2.cpu()):
      return False
  return True

def test_elementwise_3_2_broadcast_dim1_contiguous_s(shapes, dtype):
    # float
    # shape = [4,3,1000],[4,3,1050],[4,3,12800],[4,3,13600],[4,3,14000],[4,3,14800],[4,3,15200],[4,3,15600],[4,3,16000],[4,3,16800],[4,3,3200],[4,3,3400],[4,3,3500]
    #         [4,3,3700-3800-3900-4000-4200],[4,3,51200-54400-56000-59200-60800-62400-64000-67200],[4,3,800-850-875-925-950-975]
    for shape in shapes:
      a = torch.randn(shape, dtype=dtype)

      b = torch.randn(shape, dtype=dtype)

      a_c = copy.deepcopy(a)
      a_c = a_c.as_strided(shape, (1, 0, shape[0]))
      b_c = copy.deepcopy(b)
      b_c = b_c.as_strided(shape, (1, shape[0], 0))
      ref = a_c + b_c

      a_d = a.cuda()
      b_d = b.cuda()
      a_d = a_d.as_strided(shape, (1, 0, shape[0]))
      b_d = b_d.as_strided(shape, (1, shape[0], 0))
      out_d = a_d + b_d
      
      if not torch.allclose(out_d.cpu(), ref):
        return False
    return True

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  dtypes = [torch.float32, torch.float16, torch.bfloat16]
  # elementwise_3_2_broadcast_dim1_contiguous shapes
  shapes = [[24, 4608, 384], [32, 4608, 384], [21, 4608, 384]]

  if args.type == "checkin":
    odd_shapes = [[23, 4608, 384],]
    # different vec num
    vec_shapes = [[24, 4096, 64],]
    # the shape trigger y_remain, x_reamin
    ext_shapes = [[192, 4, 384],]
  else:
    odd_shapes = [[23, 4608, 384], [24, 4607, 384], [24, 4608, 385], [23, 4609, 385]]
    # different vec num
    vec_shapes = [[24, 4096, 64], [24, 4096, 128], [24, 4096, 192], [24, 4096, 256], [24, 4096, 320], [64, 4608, 384], [128, 4608, 384]]
    # the shape trigger y_remain, x_reamin
    ext_shapes = [[192, 4, 384], [192, 5, 384], [192, 7, 512]]

  shapes.extend(vec_shapes)
  shapes.extend(ext_shapes)
  shapes.extend(odd_shapes)

  for dtype in dtypes:
    if not test_elementwise_3_2_broadcast_dim1_contiguous(shapes, dtype):
      print("Error")
      exit(1)

  # elementwise_3_2_broadcast_dim1_uncontiguous shapes
  # this kernel can handle shape < 64 in contiguous dim
  # shape trigger differt z_t
  if args.type == "checkin":
    un_shapes = [[32, 1024, 2], [64, 512, 3], [64, 512, 4]]
    odd_un_shapes = [[32, 1023, 8], [31, 1024, 16]]
    # different vec num
    un_vec_shapes = [[1024, 4, 32], [1024, 4, 64]]
    # the shape trigger y_remain, x_reamin
    un_ext_shapes = [[1088, 4, 384],]
    # shape in ChatGLM
    chat_glm_shape = [[2048, 8, 32], ]
  else:
    un_shapes = [[32, 1024, 2], [64, 512, 3], [64, 512, 4], [128, 256, 6], [128, 256, 8], [21, 128, 16], 
                [1024, 8, 32], [1024, 16, 64], [512, 32, 128], [1024, 4, 256], [2048, 2, 384]]
    odd_un_shapes = [[32, 1023, 8], [31, 1024, 16], [33, 1025, 2], [1023, 17, 64], [511, 33, 128], [1029, 3, 256]]
    # different vec num
    un_vec_shapes = [[1024, 4, 32], [1024, 4, 64], [1024, 8, 128], [1024, 8, 192], [1024, 4, 256], [1024, 4, 320]]
    # the shape trigger y_remain, x_reamin
    un_ext_shapes = [[1088, 4, 384], [1088, 5, 384], [1152, 7, 512]]
    # shape in ChatGLM
    chat_glm_shape = [[2048, 8, 32], [2048, 8, 64]]

  un_shapes.extend(un_vec_shapes)
  un_shapes.extend(un_ext_shapes)
  un_shapes.extend(odd_un_shapes)

  for dtype in [torch.float16]:
    if not test_elementwise_3_2_broadcast_dim1_uncontiguous(un_shapes, dtype):
      print("Error")
      exit(1)

  for dtype in [torch.float16, torch.bfloat16]:
    if not test_elementwise_3_2_broadcast_dim1_uncontiguous(chat_glm_shape, dtype):
      print("Error")
      exit(1)

  if args.type == "checkin":
    shapes = [[4,1000,3],[4,1050,3],[4,12800,3]]
    odd_shapes = [[32,1001,3],[4,1050,3]]
    shapes.extend(odd_shapes)
  else:
    shapes = [[4,1000,3],[4,1050,3],[4,12800,3],[4,13600,3],[4,14000,3],[4,14800,3],[4,15200,3],[4,15600,3],[4,16000,3],[4,16800,3],[4,3200,3],[4,3400,3],[4,3500,3],
    [4,3700,3],[4,3800,3],[4,3900,3],[4,4000,3],[4,4200,3],[4,51200,3],[4,54400,3],[4,56000,3],[4,59200,3],[4,60800,3],[4,62400,3],[4,67200,3],[4,56000,3],[4,800,3]]
    odd_shapes = [[32,1001,3],[4,1050,3],[6,12800,3],[4,13605,3],[7,14003,3],[5,14801,3]]
    shapes.extend(odd_shapes)

  for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    if not test_elementwise_3_2_broadcast_dim1_contiguous_s(shapes, dtype):
      print("Error")
      exit(1)

  exit(0)