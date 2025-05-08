import torch
import copy
import argparse

def test_elementwise_3_2_broadcast_dim2_arg0_contiguous(shapes, dtype):
    for shape in shapes:
        a = torch.randn(shape, dtype=dtype)
        b = torch.randn(shape, dtype=dtype)

        a_c = copy.deepcopy(a)
        a_c = a_c.as_strided(shape, (1, shape[0], shape[0] * shape[1]))
        b_c = copy.deepcopy(b)
        b_c = b_c.as_strided(shape, (0, 1, 0))
        b_c_2 = b_c.as_strided(shape, (shape[1], 1, 0))
        ref = a_c + b_c
        ref_2 = a_c + b_c_2

        a_d = a.cuda()
        b_d = b.cuda()
        a_d = a_d.as_strided(shape, (1, shape[0], shape[0] * shape[1]))
        b_d = b_d.as_strided(shape, (0, 1, 0))
        b_d_2 = b_d.as_strided(shape, (shape[1], 1, 0))
        out_d = a_d + b_d
        out_d_2 = a_d + b_d_2
        if not torch.allclose(ref, out_d.cpu()) or not torch.allclose(ref_2, out_d_2.cpu()):
            return False
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="checkin", help="<checkin|daily>")

    args = parser.parse_args()

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    shapes = [[1146880, 64, 2], [143360, 32, 2], [2838528, 64, 2]]
    if args.type == "checkin":
        odd_shapes = [[1146881, 64, 2], [1146880, 63, 2]]
        # different vec num
        vec_shapes = [[24, 1024, 64],]
        # the shape trigger y_remain, x_reamin
        ext_shapes = [[192, 4, 384],]
    else:
        odd_shapes = [[1146881, 64, 2], [1146880, 63, 2], [1146880, 64, 3], [1146880, 69, 5], [143359, 32, 2], 
                      [143360, 35, 2], [143360, 32, 7], [1433657, 32, 7], [2838529, 64, 2], [2838528, 61, 2], 
                      [2838528, 64, 3], [2838527, 23, 7]]
        # different vec num
        vec_shapes = [[24, 1024, 64], [24, 1024, 128], [24, 1024, 192], [24, 1024, 256], [24, 1024, 320]]
        # the shape trigger y_remain, x_reamin
        ext_shapes = [[192, 4, 384], [192, 5, 384], [192, 7, 512]]

    shapes.extend(odd_shapes)
    shapes.extend(vec_shapes)
    shapes.extend(ext_shapes)

    for dtype in dtypes:
      if not test_elementwise_3_2_broadcast_dim2_arg0_contiguous(shapes, dtype):
          print("Error")
          exit(1)
    exit(0)