import torch

def check_close(infer_result_data, golden_data, eps=1e-4):
    diff = infer_result_data - golden_data
    diff_square = diff * diff
    infer_result_square_double = 2 * infer_result_data * infer_result_data
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))

    return result


def test_index_large_precison():
    shape_list = [(17, 18, 20, 80), (2048, 18, 20, 128), (19, 20, 21, 220,23),
                  (40, 50, 60, 70, 4), (22, 33, 44, 505, 7), (30, 40, 50, 60, 7)]

    dim_list = [1,2]
    dtype_list = [torch.half, torch.bfloat16]
    for dtype in dtype_list:
            for dim in dim_list:
                for shape in shape_list:
                    a = torch.rand(shape, device="cuda", dtype=dtype)
                    index = torch.randint(shape[dim], (shape[dim],), device="cuda")
                    src = torch.rand(shape, device="cuda", dtype=dtype)
                    out = a.index_add(dim, index, src)

                    a_c = a.detach().clone().cpu()
                    index_c = index.detach().clone().cpu()
                    src_c = src.detach().clone().cpu()
                    out_c = a_c.index_add(dim, index_c, src_c)

#                    diff = torch.max(torch.abs(out.cpu()-out_c))
                    diff = check_close(out.cpu().float(),out_c.float())
                    if diff >= 0.01:
                        print("test_index_large_precison error")
                        exit(1)
test_index_large_precison()

