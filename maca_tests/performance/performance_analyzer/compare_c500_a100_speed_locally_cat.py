c500_file = "perf_new_cat.log"
a100_file = "perf_old_cat.log"
c500_lines = []
with open(c500_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line != '':
            c500_lines.append(line)

a100_lines = []
with open(a100_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line != '':
            a100_lines.append(line)
print(len(c500_lines))
print(len(a100_lines))
assert len(c500_lines) == len(a100_lines), "c500_lines should equal to a100_lines."
c500_lines = sorted(c500_lines)
a100_lines = sorted(a100_lines)
fwd_bf16_ratio_sum = 0
fwd_bf16_ratio_count = 0
fwd_f16_ratio_sum = 0
fwd_f16_ratio_count = 0
fwd_f32_ratio_sum = 0
fwd_f32_ratio_count = 0
bwd_bf16_ratio_sum = 0
bwd_bf16_ratio_count = 0
bwd_f16_ratio_sum = 0
bwd_f16_ratio_count = 0
bwd_f32_ratio_sum = 0
bwd_f32_ratio_count = 0

contiguous_True_bf16_ratio_sum = 0
contiguous_True_bf16_ratio_count = 0
contiguous_True_f16_ratio_sum = 0
contiguous_True_f16_ratio_count = 0
contiguous_True_f32_ratio_sum = 0
contiguous_True_f32_ratio_count = 0

contiguous_False_bf16_ratio_sum = 0
contiguous_False_bf16_ratio_count = 0
contiguous_False_f16_ratio_sum = 0
contiguous_False_f16_ratio_count = 0
contiguous_False_f32_ratio_sum = 0
contiguous_False_f32_ratio_count = 0
ratio_list = []
for line1, line2 in zip(c500_lines, a100_lines):
    time1 = float(line1.split("time:")[-1])
    time2 = float(line2.split("time:")[-1])
    ratio = time1 / time2
    ratio_list.append(ratio)
    print(line1.strip())
    print(line2.strip())
    # if "contiguous_False" in line1:
    print(f"c500/a100 time ratio:{round(ratio*100, 2)}%")
    print("\n")
    if ",bfloat16," in line1 and "contiguous_True" in line1:
        contiguous_True_bf16_ratio_sum += ratio
        contiguous_True_bf16_ratio_count += 1
    elif ",float16," in line1 and "contiguous_True" in line1:
        contiguous_True_f16_ratio_sum += ratio
        contiguous_True_f16_ratio_count += 1
    elif ",float32," in line1 and "contiguous_True" in line1:
        contiguous_True_f32_ratio_sum += ratio
        contiguous_True_f32_ratio_count += 1
    elif ",bfloat16," in line1 and "contiguous_False" in line1:
        contiguous_False_bf16_ratio_sum += ratio
        contiguous_False_bf16_ratio_count += 1
    elif ",float16," in line1 and "contiguous_False" in line1:
        contiguous_False_f16_ratio_sum += ratio
        contiguous_False_f16_ratio_count += 1
    elif ",float32," in line1 and "contiguous_False" in line1:
        contiguous_False_f32_ratio_sum += ratio
        contiguous_False_f32_ratio_count += 1

if fwd_bf16_ratio_count != 0:
    print(f"fwd_bf16:{round(fwd_bf16_ratio_sum / fwd_bf16_ratio_count * 100, 2)}%")
if fwd_f16_ratio_count != 0:
    print(f"fwd_f16:{round(fwd_f16_ratio_sum / fwd_f16_ratio_count * 100, 2)}%")
if fwd_f32_ratio_count != 0:
    print(f"fwd_f32:{round(fwd_f32_ratio_sum / fwd_f32_ratio_count * 100, 2)}%")
if bwd_bf16_ratio_count != 0:
    print(f"bwd_bf16:{round(bwd_bf16_ratio_sum / bwd_bf16_ratio_count * 100, 2)}%")
if bwd_f16_ratio_count != 0:
    print(f"bwd_f16:{round(bwd_f16_ratio_sum / bwd_f16_ratio_count * 100, 2)}%")
if bwd_f32_ratio_count != 0:
    print(f"bwd_f32:{round(bwd_f32_ratio_sum / bwd_f32_ratio_count * 100, 2)}%")

if contiguous_True_bf16_ratio_count != 0:
    print(f"contiguous_True_bf16:{round(contiguous_True_bf16_ratio_sum / contiguous_True_bf16_ratio_count * 100, 2)}%")
if contiguous_True_f16_ratio_count != 0:
    print(f"contiguous_True_f16:{round(contiguous_True_f16_ratio_sum / contiguous_True_f16_ratio_count * 100, 2)}%")
if contiguous_True_f32_ratio_count != 0:
    print(f"contiguous_True_f32:{round(contiguous_True_f32_ratio_sum / contiguous_True_f32_ratio_count * 100, 2)}%")
if contiguous_False_bf16_ratio_count != 0:
    print(f"contiguous_False_bf16:{round(contiguous_False_bf16_ratio_sum / contiguous_False_bf16_ratio_count * 100, 2)}%")
if contiguous_False_f16_ratio_count != 0:
    print(f"contiguous_False_f16:{round(contiguous_False_f16_ratio_sum / contiguous_False_f16_ratio_count * 100, 2)}%")
if contiguous_False_f32_ratio_count != 0:
    print(f"contiguous_False_f32:{round(contiguous_False_f32_ratio_sum / contiguous_False_f32_ratio_count * 100, 2)}%")

print("contiguous_True_bf16_ratio_count:", contiguous_True_bf16_ratio_count)
print("contiguous_True_f16_ratio_count:", contiguous_True_f16_ratio_count)
print("contiguous_True_f32_ratio_count:", contiguous_True_f32_ratio_count)
print("contiguous_False_bf16_ratio_count:", contiguous_False_bf16_ratio_count)
print("contiguous_False_f16_ratio_count:", contiguous_False_f16_ratio_count)
print("contiguous_False_f32_ratio_count:", contiguous_False_f32_ratio_count)
print("max(ratio_list):", max(ratio_list))
print("min(ratio_list):", min(ratio_list))
print("avg ratio:",sum(ratio_list)/(len(ratio_list)))
print(sorted(ratio_list))