import torch
import os
import itertools
import importlib
import random
torch.manual_seed(0)
dtypes = [torch.bfloat16, torch.half]
for dtype in dtypes:
    shape_list = [((4, 32, 2048, 128), (4096, 128, 16384, 1)), ((1, 10, 1024, 64), (655360, 64, 640, 1))]
    for (shape, stride) in shape_list:
        query = torch.randn(shape, dtype = dtype, device = "cuda",  requires_grad = True)
        key = torch.randn(shape, dtype = dtype, device = "cuda", requires_grad = True)
        value = torch.randn(shape, dtype = dtype, device = "cuda",  requires_grad = True)
        query_layer_d = query.as_strided(shape, stride)
        key_layer_d = key.as_strided(shape, stride)
        value_layer_d = value.as_strided(shape, stride)
        output = torch.nn.functional.scaled_dot_product_attention(query_layer_d, key_layer_d, value_layer_d, is_causal = True)
        output_golden = torch.nn.functional._scaled_dot_product_attention(query_layer_d, key_layer_d, value_layer_d, is_causal = True)
        # scaled_dot_product_attention forward
        print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output - output_golden)))
        if dtype == torch.half:
            rtol_value = 1e-2
            atol_value = 1e-2
        else:
            rtol_value = 1e-1
            atol_value = 1e-1
        if not torch.allclose(output, output_golden, rtol=rtol_value, atol=atol_value):
            exit(1)




batch_size_list = [2]
nheads_list = [77]
seq_length_list = [1024]
head_dim_list = [64, 192]
dtypes = [torch.half, torch.bfloat16]
use_attn_mask_list = [True, False]
attn_mask_bool_list = [True, False]
attn_mask_dim_list = [1, 2, 3, 4]
for batch_size, nheads, seq_length, head_dim, dtype, use_attn_mask, attn_mask_bool, attn_mask_dim in itertools.product(batch_size_list, nheads_list, seq_length_list, head_dim_list, dtypes, use_attn_mask_list, attn_mask_bool_list, attn_mask_dim_list):
    query = torch.randn(batch_size, nheads, seq_length, head_dim, dtype=dtype).cuda().requires_grad_()
    key = torch.randn(batch_size, nheads, seq_length, head_dim, dtype=dtype).cuda().requires_grad_()
    value = torch.randn(batch_size, nheads, seq_length, head_dim, dtype=dtype).cuda().requires_grad_()
    if use_attn_mask:
        attn_weight_shape = [batch_size, nheads, seq_length, seq_length]
        attn_mask_shape = attn_weight_shape[-attn_mask_dim::]
        if attn_mask_bool:
            attn_mask = torch.rand(*attn_mask_shape).cuda()
            attn_mask = attn_mask >= 0.5
        else:
            attn_mask = torch.rand(*attn_mask_shape, dtype=query.dtype).cuda()
    else:
        attn_mask = None
    print(batch_size, nheads, seq_length, head_dim, dtype, attn_mask_shape)

    query_clone = query.detach().clone().requires_grad_()
    key_clone = key.detach().clone().requires_grad_()
    value_clone = value.detach().clone().requires_grad_()

    result_math = torch.nn.functional._scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask)
    result_math.sum().backward()
    result_flash = torch.nn.functional.scaled_dot_product_attention(query_clone, key_clone, value_clone, dropout_p=0.0, is_causal=False, attn_mask=attn_mask)
    result_flash.sum().backward()

    #calc err
    if dtype == torch.half:
        rtol_value = 1e-2
        atol_value = 1e-2
    else:
        rtol_value = 1e-1
        atol_value = 1e-1
    forward_result = torch.allclose(result_math.float(), result_flash.float() ,rtol = rtol_value, atol = atol_value)
    print("result_math.std:", result_math.std())
    print("result_flash.std:", result_flash.std())
    print("result_math.mean():", result_math.mean())
    print("result_flash.mean:", result_flash.mean())
    print("forward_result:", forward_result)
    query_grad_result = torch.allclose(query.grad.float(), query_clone.grad.float() , rtol = rtol_value, atol = atol_value)
    print("query.grad.std:", query.grad.std())
    print("query_clone.grad.std:", query_clone.grad.std())
    print("query.grad.mean():", query.grad.mean())
    print("query_clone.grad.mean():", query_clone.grad.mean())
    print("query_grad_result:", query_grad_result)
    key_grad_result = torch.allclose(key.grad.float(), key_clone.grad.float() , rtol = rtol_value, atol = atol_value)
    print("key.grad.std:", key.grad.std())
    print("key_clone.grad.std:", key_clone.grad.std())
    print("key.grad.mean():", key.grad.mean())
    print("key_clone.grad.mean():", key_clone.grad.mean())
    print("key_grad_result:", key_grad_result)
    value_grad_result = torch.allclose(value.grad.float(), value_clone.grad.float() , rtol = rtol_value, atol = atol_value)
    print("value.grad.std:", value.grad.std())
    print("value_clone.grad.std:", value_clone.grad.std())
    print("value.grad.mean():", value.grad.mean())
    print("value_clone.grad.mean():", value_clone.grad.mean())
    print("value_grad_result:", value_grad_result)
    assert (forward_result and query_grad_result and key_grad_result and value_grad_result)

#add tests for not contiguous attn_mask
def construct_non_contiguous_attn_mask(attn_mask):
    non_contiguous_attn_mask_list = []
    shape = attn_mask.shape
    length = len(shape)
    if length == 1:
        non_contiguous_attn_mask = attn_mask[0].expand(shape)
        return non_contiguous_attn_mask
    elif length == 2:
        non_contiguous_attn_mask = attn_mask[0,:].unsqueeze(0).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        non_contiguous_attn_mask = attn_mask[:,0].unsqueeze(1).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        return random.choice(non_contiguous_attn_mask_list)
    elif length == 3:
        non_contiguous_attn_mask = attn_mask[0,:,:].unsqueeze(0).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        non_contiguous_attn_mask = attn_mask[:,0,:].unsqueeze(1).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        non_contiguous_attn_mask = attn_mask[:,:,0].unsqueeze(2).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        return random.choice(non_contiguous_attn_mask_list)
    elif length == 4:
        non_contiguous_attn_mask = attn_mask[0,:,:,:].unsqueeze(0).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        non_contiguous_attn_mask = attn_mask[:,0,:,:].unsqueeze(1).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        non_contiguous_attn_mask = attn_mask[:,:,0,:].unsqueeze(2).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        non_contiguous_attn_mask = attn_mask[:,:,:,0].unsqueeze(3).expand(shape)
        non_contiguous_attn_mask_list.append(non_contiguous_attn_mask)
        return random.choice(non_contiguous_attn_mask_list)

batch_size_list = [1, 7]
nheads_list = [77,133]
seq_length_list = [133, 511,]
head_dim_list = [63, 113, 139]
dtypes = [torch.half, torch.bfloat16]
attn_mask_dim_list = [1, 2, 3, 4]
for batch_size, nheads, seq_length, head_dim, dtype, attn_mask_dim in itertools.product(batch_size_list, nheads_list, seq_length_list, head_dim_list, dtypes, attn_mask_dim_list):
    query = torch.randn(batch_size, nheads, seq_length, head_dim, dtype=dtype).cuda().requires_grad_()
    key = torch.randn(batch_size, nheads, seq_length, head_dim, dtype=dtype).cuda().requires_grad_()
    value = torch.randn(batch_size, nheads, seq_length, head_dim, dtype=dtype).cuda().requires_grad_()
    attn_weight_shape = [batch_size, nheads, seq_length, seq_length]
    attn_mask_shape = attn_weight_shape[-attn_mask_dim::]
    attn_mask = torch.rand(*attn_mask_shape, dtype=query.dtype).cuda()
    # print(attn_mask_shape, attn_mask.stride())
    # print(attn_mask)
    attn_mask = construct_non_contiguous_attn_mask(attn_mask)
    assert(not attn_mask.is_contiguous())
    print(batch_size, nheads, seq_length, head_dim, dtype, attn_mask_dim, attn_mask_shape, attn_mask.stride())
    query_clone = query.detach().clone().requires_grad_()
    key_clone = key.detach().clone().requires_grad_()
    value_clone = value.detach().clone().requires_grad_()

    result_math = torch.nn.functional._scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask)
    result_math.sum().backward()
    result_flash = torch.nn.functional.scaled_dot_product_attention(query_clone, key_clone, value_clone, dropout_p=0.0, is_causal=False, attn_mask=attn_mask)
    result_flash.sum().backward()

    #calc err
    if dtype == torch.half:
        rtol_value = 1e-3
        atol_value = 1
    else:
        rtol_value = 1e-3
        atol_value = 1
    forward_result = torch.allclose(result_math.float(), result_flash.float() ,rtol = rtol_value, atol = atol_value)
    print("result_math.std:", result_math.std())
    print("result_flash.std:", result_flash.std())
    print("result_math.mean():", result_math.mean())
    print("result_flash.mean:", result_flash.mean())
    print("forward_result:", forward_result)
    query_grad_result = torch.allclose(query.grad.float(), query_clone.grad.float() , rtol = rtol_value, atol = atol_value)
    print("query.grad.std:", query.grad.std())
    print("query_clone.grad.std:", query_clone.grad.std())
    print("query.grad.mean():", query.grad.mean())
    print("query_clone.grad.mean():", query_clone.grad.mean())
    print("query_grad_result:", query_grad_result)
    key_grad_result = torch.allclose(key.grad.float(), key_clone.grad.float() , rtol = rtol_value, atol = atol_value)
    print("key.grad.std:", key.grad.std())
    print("key_clone.grad.std:", key_clone.grad.std())
    print("key.grad.mean():", key.grad.mean())
    print("key_clone.grad.mean():", key_clone.grad.mean())
    print("key_grad_result:", key_grad_result)
    value_grad_result = torch.allclose(value.grad.float(), value_clone.grad.float() , rtol = rtol_value, atol = atol_value)
    print("value.grad.std:", value.grad.std())
    print("value_clone.grad.std:", value_clone.grad.std())
    print("value.grad.mean():", value.grad.mean())
    print("value_clone.grad.mean():", value_clone.grad.mean())
    print("value_grad_result:", value_grad_result)
    assert (forward_result and query_grad_result and key_grad_result and value_grad_result)