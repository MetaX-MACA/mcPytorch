import torch


def test_foreach_abs(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_abs([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_abs([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_add(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_add([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_add([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_atan(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_atan([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_atan([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_ceil(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_ceil([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_ceil([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_clamp_max(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_clamp_max([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_clamp_max([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_clamp_min(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_clamp_min([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_clamp_min([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_cos(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_cos([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_cos([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_cosh(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_cosh([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_cosh([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_div(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_div([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_div([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu(), atol=1e-7):
            return False
    return True

def test_foreach_erf(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_erf([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_erf([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu(), atol=1e-6):
            return False
    return True

def test_foreach_erfc(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_erfc([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_erfc([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_exp(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_exp([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_exp([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_expm1(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_expm1([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_expm1([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_floor(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_floor([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_floor([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_frac(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_frac([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_frac([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_maximum(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_maximum([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_maximum([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_minimum(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_minimum([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_minimum([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

# test double, float, half, bfloat16
def test_foreach_mul(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_mul([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_mul([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_neg(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_neg([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_neg([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_reciprocal(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_reciprocal([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_reciprocal([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_round(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_round([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_round([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_sigmoid(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_sigmoid([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_sigmoid([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_sin(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_sin([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_sin([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_sinh(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_sinh([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_sinh([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_sub(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_sub([a, b], 2)

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_sub([ad, bd], 2)

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_tan(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_tan([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_tan([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_trunc(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_trunc([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_trunc([ad, bd])

    for i in range(len(ref)):
        if not torch.allclose(ref[i], out[i].cpu()):
            return False
    return True

def test_foreach_zero(dtype, shape):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    ref = torch._foreach_zero_([a, b])

    ad = a.cuda()
    bd = b.cuda()
    out = torch._foreach_zero_([ad, bd])
    if torch.all(ref[0] == 0) and torch.all(ref[1] == 0) and torch.all(out[0] == 0) and torch.all(out[1] == 0):
        return True
    else:
        return False

# float32, float16
def test_amp_foreach_non_finite_check_and_unscale(dtype):
    shape = [24, 1024]
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)

    # test no inf
    found_inf = torch.tensor(0, dtype=torch.float32, device="cuda")
    scale = torch.tensor(2, dtype=torch.float32)
    a_d = a.cuda()
    b_d = b.cuda()
    scale_d = scale.cuda()
    ref_a = a * scale
    ref_b = b * scale
    torch._amp_foreach_non_finite_check_and_unscale_([a_d, b_d], found_inf, scale_d)
    if not torch.allclose(ref_a, a_d.cpu()) or not torch.allclose(ref_b, b_d.cpu()) or not torch.allclose(torch.tensor(0, dtype=torch.float32), found_inf.cpu()):
        return False
    
    # test inf
    a[0][0] = torch.inf
    b[1][0] = torch.inf
    a_d = a.cuda()
    b_d = b.cuda()
    ref_a = a * scale
    ref_b = b * scale
    torch._amp_foreach_non_finite_check_and_unscale_([a_d, b_d], found_inf, scale_d)
    if not torch.allclose(ref_a, a_d.cpu()) or not torch.allclose(ref_b, b_d.cpu()) or not torch.allclose(torch.tensor(1, dtype=torch.float32), found_inf.cpu()):
        return False

    return True

if __name__ == "__main__":
    for dtype in [torch.double, torch.float32, torch.float16, torch.bfloat16, ]:
        for shape in [[24, 1024], [24, 38, 1024], ]:
            if not test_foreach_mul(dtype, shape):
                print("test_foreach_mul opt fail!")
                exit(1)
            if not test_foreach_abs(dtype, shape):
                print("test_foreach_abs opt fail!")
                exit(1)
            if not test_foreach_add(dtype, shape):
                print("test_foreach_add opt fail!")
                exit(1)
            if not test_foreach_div(dtype, shape):
                print("test_foreach_div opt fail!")
                exit(1)
            if not test_foreach_frac(dtype, shape):
                print("test_foreach_frac opt fail!")
                exit(1)
            if not test_foreach_neg(dtype, shape):
                print("test_foreach_neg opt fail!")
                exit(1)
            if not test_foreach_reciprocal(dtype, shape):
                print("test_foreach_reciprocal opt fail!")
                exit(1)
            if not test_foreach_zero(dtype, shape):
                print("test_foreach_zero opt fail!")
                exit(1)

    for dtype in [torch.double, torch.float32,]:
        for shape in [[24, 1024], [24, 38, 1024], ]:
            if not test_foreach_atan(dtype, shape):
                print("test_foreach_atan opt fail!")
                exit(1)
            if not test_foreach_ceil(dtype, shape):
                print("test_foreach_ceil opt fail!")
                exit(1)
            if not test_foreach_clamp_max(dtype, shape):
                print("test_foreach_clamp_max opt fail!")
                exit(1)
            if not test_foreach_clamp_min(dtype, shape):
                print("test_foreach_clamp_min opt fail!")
                exit(1)
            if not test_foreach_cos(dtype, shape):
                print("test_foreach_cos opt fail!")
                exit(1)
            if not test_foreach_cosh(dtype, shape):
                print("test_foreach_cosh opt fail!")
                exit(1)
            if not test_foreach_erf(dtype, shape):
                print("test_foreach_erf opt fail!")
                exit(1)
            if not test_foreach_erfc(dtype, shape):
                print("test_foreach_erfc opt fail!")
                exit(1)
            if not test_foreach_exp(dtype, shape):
                print("test_foreach_exp opt fail!")
                exit(1)
            if not test_foreach_expm1(dtype, shape):
                print("test_foreach_expm1 opt fail!")
                exit(1)
            if not test_foreach_floor(dtype, shape):
                print("test_foreach_floor opt fail!")
                exit(1)
            if not test_foreach_maximum(dtype, shape):
                print("test_foreach_maximum opt fail!")
                exit(1)
            if not test_foreach_minimum(dtype, shape):
                print("test_foreach_minimum opt fail!")
                exit(1)
            if not test_foreach_round(dtype, shape):
                print("test_foreach_round opt fail!")
                exit(1)
            if not test_foreach_sigmoid(dtype, shape):
                print("test_foreach_sigmoid opt fail!")
                exit(1)
            if not test_foreach_sin(dtype, shape):
                print("test_foreach_sin opt fail!")
                exit(1)
            if not test_foreach_sinh(dtype, shape):
                print("test_foreach_sinh opt fail!")
                exit(1)
            if not test_foreach_sub(dtype, shape):
                print("test_foreach_sub opt fail!")
                exit(1)
            if not test_foreach_tan(dtype, shape):
                print("test_foreach_tan opt fail!")
                exit(1)
            if not test_foreach_trunc(dtype, shape):
                print("test_foreach_trunc opt fail!")
                exit(1)

    for dtype in [torch.float32, torch.float16, ]:
        if not test_amp_foreach_non_finite_check_and_unscale(dtype):
            print("test_amp_foreach_non_finite_check_and_unscale opt fail!")
            exit(1)
    exit(0)
