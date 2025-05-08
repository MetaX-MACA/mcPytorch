c500_file = "c500.log"
a100_file = "a100.log"
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

log_fwd_bf16_ratio_sum = 0
log_fwd_bf16_ratio_count = 0
log_fwd_f16_ratio_sum = 0
log_fwd_f16_ratio_count = 0
log_fwd_f32_ratio_sum = 0
log_fwd_f32_ratio_count = 0
log_bwd_bf16_ratio_sum = 0
log_bwd_bf16_ratio_count = 0
log_bwd_f16_ratio_sum = 0
log_bwd_f16_ratio_count = 0
log_bwd_f32_ratio_sum = 0
log_bwd_f32_ratio_count = 0
ratio_list = []
for line1, line2 in zip(c500_lines, a100_lines):
    time1 = float(line1.split("time:")[-1])
    time2 = float(line2.split("time:")[-1])
    ratio = time1 / time2
    ratio_list.append(ratio)
    print(line1.strip())
    print(line2.strip())
    print(f"c500/a100 time ratio:{round(ratio*100, 2)}%")
    print("\n")
    if "log" not in line1:
        if ",bfloat16," in line1 and "fwd" in line1:
            fwd_bf16_ratio_sum += ratio
            fwd_bf16_ratio_count += 1
        elif ",float16," in line1 and "fwd" in line1:
            fwd_f16_ratio_sum += ratio
            fwd_f16_ratio_count += 1
        elif ",float32," in line1 and "fwd" in line1:
            fwd_f32_ratio_sum += ratio
            fwd_f32_ratio_count += 1
        elif ",bfloat16," in line1 and "bwd" in line1:
            bwd_bf16_ratio_sum += ratio
            bwd_bf16_ratio_count += 1
        elif ",float16," in line1 and "bwd" in line1:
            bwd_f16_ratio_sum += ratio
            bwd_f16_ratio_count += 1
        elif ",float32," in line1 and "bwd" in line1:
            bwd_f32_ratio_sum += ratio
            bwd_f32_ratio_count += 1
    else:
        if ",bfloat16," in line1 and "fwd" in line1:
            log_fwd_bf16_ratio_sum += ratio
            log_fwd_bf16_ratio_count += 1
        elif ",float16," in line1 and "fwd" in line1:
            log_fwd_f16_ratio_sum += ratio
            log_fwd_f16_ratio_count += 1
        elif ",float32," in line1 and "fwd" in line1:
            log_fwd_f32_ratio_sum += ratio
            log_fwd_f32_ratio_count += 1
        elif ",bfloat16," in line1 and "bwd" in line1:
            log_bwd_bf16_ratio_sum += ratio
            log_bwd_bf16_ratio_count += 1
        elif ",float16," in line1 and "bwd" in line1:
            log_bwd_f16_ratio_sum += ratio
            log_bwd_f16_ratio_count += 1
        elif ",float32," in line1 and "bwd" in line1:
            log_bwd_f32_ratio_sum += ratio
            log_bwd_f32_ratio_count += 1

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

if log_fwd_bf16_ratio_count != 0:
    print(f"log_fwd_bf16:{round(log_fwd_bf16_ratio_sum / log_fwd_bf16_ratio_count * 100, 2)}%")
if log_fwd_f16_ratio_count != 0:
    print(f"log_fwd_f16:{round(log_fwd_f16_ratio_sum / log_fwd_f16_ratio_count * 100, 2)}%")
if log_fwd_f32_ratio_count != 0:
    print(f"log_fwd_f32:{round(log_fwd_f32_ratio_sum / log_fwd_f32_ratio_count * 100, 2)}%")
if log_bwd_bf16_ratio_count != 0:
    print(f"log_bwd_bf16:{round(log_bwd_bf16_ratio_sum / log_bwd_bf16_ratio_count * 100, 2)}%")
if log_bwd_f16_ratio_count != 0:
    print(f"log_bwd_f16:{round(log_bwd_f16_ratio_sum / log_bwd_f16_ratio_count * 100, 2)}%")
if log_bwd_f32_ratio_count != 0:
    print(f"log_bwd_f32:{round(log_bwd_f32_ratio_sum / log_bwd_f32_ratio_count * 100, 2)}%")

print("fwd_bf16_ratio_count:", fwd_bf16_ratio_count)
print("fwd_f16_ratio_count:", fwd_f16_ratio_count)
print("fwd_f32_ratio_count:", fwd_f32_ratio_count)
print("bwd_bf16_ratio_count:", bwd_bf16_ratio_count)
print("bwd_f16_ratio_count:", bwd_f16_ratio_count)
print("bwd_f32_ratio_count:", bwd_f32_ratio_count)
print("log_fwd_bf16_ratio_count:", log_fwd_bf16_ratio_count)
print("log_fwd_f16_ratio_count:", log_fwd_f16_ratio_count)
print("log_fwd_f32_ratio_count:", log_fwd_f32_ratio_count)
print("log_bwd_bf16_ratio_count:", log_bwd_bf16_ratio_count)
print("log_bwd_f16_ratio_count:", log_bwd_f16_ratio_count)
print("log_bwd_f32_ratio_count:", log_bwd_f32_ratio_count)
print("max(ratio_list):", max(ratio_list))
print("min(ratio_list):", min(ratio_list))
print("avg ratio:",sum(ratio_list)/(len(ratio_list)))
print(sorted(ratio_list))