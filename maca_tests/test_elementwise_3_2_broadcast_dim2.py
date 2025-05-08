import torch
import copy
import argparse

def test_elementwise_3_2_broadcast_dim2_contiguous(shapes, dtype):
  # out: dtype, shape=[s0,s1,s2], stride=[1,s0,s0*s1]
  # arg0: dtype, shape=[s0,s1,s2], stride=[1,s0,s0*s1]
  # arg1: dtype (broadcast), shape=[s0,s1,s2], stride=[1,!=0, 0]
  for shape in shapes:
    a = torch.randn(shape, dtype=dtype)
    a = a.transpose(0, 2)

    b = torch.randn(shape, dtype=dtype)
    b = b.transpose(0, 2)

    b_c = copy.deepcopy(b)
    b_c = b_c.as_strided(b_c.shape, (1, b_c.shape[0] * 2, 0))
    ref = a + b_c

    a_d = a.cuda()
    b_d = b.cuda()
    b_d = b_d.as_strided(b_d.shape, (1, b_d.shape[0] * 2, 0))
    out = a_d + b_d
    if not torch.allclose(ref, out.cpu()):
      return False
  return True

def test_elementwise_3_2_broadcast_dim2_uncontiguous(shapes, dtype):
  for shape in shapes:
    a = torch.randn(shape[0] * shape[1], 2, shape[0], dtype=dtype)
    a = a.transpose(0, 2)

    b = torch.randn(shape, dtype=dtype)
    b = b.transpose(0, 2)

    a_c = copy.deepcopy(a)
    b_c = copy.deepcopy(b)
    a_c = a_c.as_strided(b.shape, (2, b.shape[2], b.shape[1] * b.shape[2]))
    b_c = b_c.as_strided(b.shape, (2, b.shape[0] * 2, 0))
    ref = a_c * b_c

    a_c_2 = copy.deepcopy(a)
    b_c_2 = copy.deepcopy(b)
    a_c_2 = a_c_2.as_strided(b.shape, (2, b.shape[2], b.shape[1] * b.shape[2]))
    b_c_2 = b_c_2.as_strided(b.shape, (2, b.shape[0], 0))
    ref_2 = a_c_2 * b_c_2

    a_d = copy.deepcopy(a).cuda()
    b_d = copy.deepcopy(b).cuda()
    a_d = a_d.as_strided(b.shape, (2, b.shape[2], b.shape[1] * b.shape[2]))
    b_d = b_d.as_strided(b.shape, (2, b.shape[0] * 2, 0))
    out = a_d * b_d

    a_d_2 = copy.deepcopy(a).cuda()
    b_d_2 = copy.deepcopy(b).cuda()
    a_d_2 = a_d_2.as_strided(b.shape, (2, b.shape[2] , b.shape[1] * b.shape[2]))
    b_d_2 = b_d_2.as_strided(b.shape, (2, b.shape[0], 0))
    out_2 = a_d_2 * b_d_2

    if not torch.allclose(ref, out.cpu()) or not torch.allclose(ref_2, out_2.cpu()):
      return False
  return True

def test_elementwise_3_2_broadcast_dim2_contiguous_s(shapes, dtype):
    # float
    # shape = [2,217413,3], [2,1000,2->24], [2,159882,2-4-5-6],[2,185460,2],[2,191835,14],[2,204624,2-3-4-6-7],[2,211038,8],[2,217413,2->15],[2,217413,19->23],[2,223827,2-12-13]
    #          [2,225603,4-14],[2,236616,2-5],[2,242991,2->10],[2,242991,12-13-14-16-18-19],[2,249405,3],[2,255780,5-12],[2,257796,2-3-6-16]
    # shape no-opt opt a100
    # [2,217413,3] 92.5us 52.6us
    # [2,2,159882] 54us 35us
    # 
    # shape = [2,4,225603]
    # shape = [4,13600,3]
    for shape in shapes:
      a = torch.randn(shape, dtype=dtype)

      b = torch.randn(shape, dtype=dtype)

      a_c = copy.deepcopy(a)
      a_c = a_c.as_strided(shape, (1, shape[0] * 2, 0))
      b_c = copy.deepcopy(b)
      b_c = b_c.as_strided(shape, (1, 0, shape[0] * 2))
      ref_max = torch.maximum(a_c, b_c)
      ref_min = torch.minimum(a_c, b_c)

      a_d = a.cuda()
      b_d = b.cuda()
      a_d = a_d.as_strided(shape, (1, shape[0] * 2, 0))
      b_d = b_d.as_strided(shape, (1, 0, shape[0] * 2))
      out_d_max = torch.maximum(a_d, b_d)
      out_d_min = torch.minimum(a_d, b_d)
      
      if not torch.allclose(ref_max, out_d_max.cpu()) or not torch.allclose(ref_min, out_d_min.cpu()):
        return False
    return True

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  dtypes = [torch.float32, torch.float16, torch.bfloat16]

  if args.type == "checkin":
    # elementwise_3_2_broadcast_dim2_contiguous shapes
    shapes = [[24, 4608, 384], [32, 4608, 384]]
    odd_shapes = [[31, 4608, 384],]
    # different vec num
    vec_shapes = [[24, 1024, 64],]
    # the shape trigger y_remain, x_reamin
    ext_shapes = [[192, 4, 384],]
  else:
    # elementwise_3_2_broadcast_dim2_contiguous shapes
    shapes = [[24, 4608, 384], [32, 4608, 384], [64, 4608, 384], [128, 4608, 384], [21, 4608, 384]]
    odd_shapes = [[31, 4608, 384], [32, 4607, 384], [32, 4608, 383], [33, 4607, 384]]
    # different vec num
    vec_shapes = [[24, 1024, 64], [24, 1024, 128], [24, 1024, 192], [24, 1024, 256], [24, 1024, 320]]
    # the shape trigger y_remain, x_reamin
    ext_shapes = [[192, 4, 384], [192, 5, 384], [192, 7, 512]]

  shapes.extend(vec_shapes)
  shapes.extend(ext_shapes)
  shapes.extend(odd_shapes)

  for dtype in dtypes:
    if not test_elementwise_3_2_broadcast_dim2_contiguous(shapes, dtype):
      print("Error")
      exit(1)

  if args.type == "checkin":
    # elementwise_3_2_broadcast_dim2_uncontiguous shapes
    # this kernel can handle shape < 64 in contiguous dim
    # shape trigger differt z_t
    un_shapes = [[32, 1024, 2], [64, 512, 3], [64, 512, 4]]
    odd_un_shapes = [[63, 128, 6], ]
    # different vec num
    un_vec_shapes = [[16, 1024, 32], ]
    # the shape trigger y_remain, x_reamin
    un_ext_shapes = [[192, 4, 384], ]
    # shape in ChatGLM
    chat_glm_shape = [[128, 2048, 32], [128, 2048, 8]]
  else:
    un_shapes = [[32, 1024, 2], [64, 512, 3], [64, 512, 4], [128, 256, 6], [128, 256, 8], [21, 128, 16], 
                [16, 1024, 32], [32, 512, 64], [64, 256, 128], [128, 128, 256], [128, 128, 384], [2304, 64, 6]]
    odd_un_shapes = [[63, 128, 6], [64, 129, 4], [50, 128, 5], [64, 129, 4], [63, 129, 4], [17, 1021, 32], [33, 511, 64], [65, 255, 128]]
    # different vec num
    un_vec_shapes = [[16, 1024, 32], [16, 512, 64], [24, 128, 128], [24, 256, 192], [16, 128, 256], [8, 128, 320]]
    # the shape trigger y_remain, x_reamin
    un_ext_shapes = [[192, 4, 384], [192, 5, 384], [192, 7, 512]]
    # shape in ChatGLM
    chat_glm_shape = [[128, 2048, 64], [128, 2048, 32], [128, 2048, 8]]

  un_shapes.extend(un_vec_shapes)
  un_shapes.extend(un_ext_shapes)
  un_shapes.extend(odd_un_shapes)

  for dtype in dtypes:
    if not test_elementwise_3_2_broadcast_dim2_uncontiguous(un_shapes, dtype):
      print("Error")
      exit(1)

  for dtype in [torch.float16, torch.bfloat16]:
    if not test_elementwise_3_2_broadcast_dim2_uncontiguous(chat_glm_shape, dtype):
      print("Error")
      exit(1)

  if args.type == "daily":
    shapes = []
    for s0 in range(2, 3):
      for s1 in [217413,1000,159882,185460,191835,204624,211038,217413,223827,225603,236616,242991,249405,255780,257796]:
        for s2 in range(2, 25):
          shape = [s0, s2, s1]
          shapes.append(shape)
    for dtype in dtypes:
      if not test_elementwise_3_2_broadcast_dim2_contiguous_s(shapes, dtype):
        print("Error")
        exit(1)

  exit(0)