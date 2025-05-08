import torch
import torch.nn.functional as F
import os


def test_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def test_attention_total():
    query_1 = torch.rand(2, 4096, 5, 64, dtype = torch.float16).cuda().contiguous()
    key_1 = torch.rand(2, 4096, 5, 64, dtype = torch.float16).cuda().contiguous()
    value_1 = torch.rand(2, 4096, 5, 64, dtype = torch.float16).cuda().contiguous()
    query_1.transpose_(1,2)
    key_1.transpose_(1,2)
    value_1.transpose_(1,2)

    query_2 = torch.rand(2, 1024, 10,64, dtype = torch.float16).cuda().contiguous()
    key_2 = torch.rand(2, 1024,10, 64, dtype = torch.float16).cuda().contiguous()
    value_2 = torch.rand(2, 1024, 10, 64, dtype = torch.float16).cuda().contiguous()
    query_2.transpose_(1,2)
    key_2.transpose_(1,2)
    value_2.transpose_(1,2)

    query_3 = torch.rand(2, 256, 20, 64, dtype = torch.float16).cuda().contiguous()
    key_3 = torch.rand(2, 256, 20, 64, dtype = torch.float16).cuda().contiguous()
    value_3 = torch.rand(2, 256, 20, 64, dtype = torch.float16).cuda().contiguous()
    query_3.transpose_(1,2)
    key_3.transpose_(1,2)
    value_3.transpose_(1,2)

    query_4 = torch.rand(2, 1024, 10, 64, dtype = torch.float16).cuda().contiguous()
    key_4 = torch.rand(2, 77, 10, 64, dtype = torch.float16).cuda().contiguous()
    value_4 = torch.rand(2, 77, 10, 64, dtype = torch.float16).cuda().contiguous()
    query_4.transpose_(1,2)
    key_4.transpose_(1,2)
    value_4.transpose_(1,2)

    query_5 = torch.rand(2, 256, 20, 64, dtype = torch.float16).cuda().contiguous()
    key_5 = torch.rand(2, 77, 20, 64, dtype = torch.float16).cuda().contiguous()
    value_5 = torch.rand(2, 77, 20, 64, dtype = torch.float16).cuda().contiguous()
    query_5.transpose_(1,2)
    key_5.transpose_(1,2)
    value_5.transpose_(1,2)

    query_6 = torch.rand(2, 64, 20, 64, dtype = torch.float16).cuda().contiguous()
    key_6 = torch.rand(2, 77, 20, 64, dtype = torch.float16).cuda().contiguous()
    value_6 = torch.rand(2, 77, 20, 64, dtype = torch.float16).cuda().contiguous()
    query_6.transpose_(1,2)
    key_6.transpose_(1,2)
    value_6.transpose_(1,2)

    query_7 = torch.rand(2, 64, 20,  64, dtype = torch.float16).cuda().contiguous()
    key_7 = torch.rand(2, 64, 20, 64, dtype = torch.float16).cuda().contiguous()
    value_7 = torch.rand(2, 64, 20, 64, dtype = torch.float16).cuda().contiguous()
    query_7.transpose_(1,2)
    key_7.transpose_(1,2)
    value_7.transpose_(1,2)

    query_8 = torch.rand(2, 4096, 5, 64, dtype = torch.float16).cuda().contiguous()
    key_8 = torch.rand(2, 77, 5, 64, dtype = torch.float16).cuda().contiguous()
    value_8 = torch.rand(2, 77, 5, 64, dtype = torch.float16).cuda().contiguous()
    query_8.transpose_(1,2)
    key_8.transpose_(1,2)
    value_8.transpose_(1,2)

    #sdxl shape
    query_9 = torch.rand(2, 4096, 10, 64, dtype = torch.float16).cuda().contiguous()
    key_9 = torch.rand(2, 4096, 10, 64, dtype = torch.float16).cuda().contiguous()
    value_9 = torch.rand(2, 4096, 10, 64, dtype = torch.float16).cuda().contiguous()
    query_9.transpose_(1,2)
    key_9.transpose_(1,2)
    value_9.transpose_(1,2)

    query_10 = torch.rand(2, 4096, 10, 64, dtype = torch.float16).cuda().contiguous()
    key_10 = torch.rand(2, 77, 10, 64, dtype = torch.float16).cuda().contiguous()
    value_10 = torch.rand(2, 77, 10, 64, dtype = torch.float16).cuda().contiguous()
    query_10.transpose_(1,2)
    key_10.transpose_(1,2)
    value_10.transpose_(1,2)

    query_11 = torch.rand(2, 1024, 20, 64, dtype = torch.float16).cuda().contiguous()
    key_11 = torch.rand(2, 1024, 20, 64, dtype = torch.float16).cuda().contiguous()
    value_11 = torch.rand(2, 1024, 20, 64, dtype = torch.float16).cuda().contiguous()
    query_11.transpose_(1,2)
    key_11.transpose_(1,2)
    value_11.transpose_(1,2)

    query_12 = torch.rand(2, 1024, 20, 64, dtype = torch.float16).cuda().contiguous()
    key_12 = torch.rand(2, 77, 20, 64, dtype = torch.float16).cuda().contiguous()
    value_12 = torch.rand(2, 77, 20, 64, dtype = torch.float16).cuda().contiguous()
    query_12.transpose_(1,2)
    key_12.transpose_(1,2)
    value_12.transpose_(1,2)

    query_13 = torch.rand(1, 16384, 1, 512, dtype = torch.float16).cuda().contiguous()
    key_13 = torch.rand(1, 16384, 1, 512, dtype = torch.float16).cuda().contiguous()
    value_13 = torch.rand(1, 16384, 1, 512, dtype = torch.float16).cuda().contiguous()
    query_13.transpose_(1,2)
    key_13.transpose_(1,2)
    value_13.transpose_(1,2)

    input_list = [[query_1, key_1, value_1], [query_2, key_2, value_2], [query_3, key_3, value_3], [query_4, key_4, value_4], [query_5, key_5, value_5], [query_6, key_6, value_6], [query_7, key_7, value_7], [query_8, key_8, value_8],\
                  [query_9, key_9, value_9], [query_10, key_10, value_10], [query_11, key_11, value_11], [query_12, key_12, value_12], [query_13, key_13, value_13]]

    for input in input_list:
        result_mha = test_attention(input[0], input[1], input[2])
        result_math = test_attention(input[0].cpu().float(), input[1].cpu().float(), input[2].cpu().float())
        if(torch.max(torch.abs(result_mha.cpu() - result_math)) <= 0.0006):
            print("input pass:", input[0].shape, input[1].shape,input[2].shape, torch.max(torch.abs(result_mha.cpu() - result_math)))
            pass
        else:
            print("max error:", torch.max(torch.abs(result_mha.cpu() - result_math)))
            print("result_mha.mean():", result_mha.mean())
            print("result_mha.std():", result_mha.std())
            print("result_math.mean():", result_math.mean())
            print("result_math.std():", result_math.std())
            exit(1)
    exit(0)



if __name__ == "__main__":
    test_attention_total()
