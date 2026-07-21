#!/bin/bash
# set -x
current_path=$(pwd)
rm -rf  $current_path/dump_graphs
mkdir $current_path/dump_graphs



MODEL_LIST="../dynamo/timm_models_list.txt"

if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: model list file: '$MODEL_LIST' not exist!"
    exit 1
fi

# model_name from $MODEL_LIST
if [ -z "$1" ]; then
    model_name="run_all"
else
    model_name="$1"
fi

# 'inference' or 'training'
if [ -z "$2" ]; then
    test_mode="training"
else
    test_mode="$2"
fi

# 'performance' or 'accuracy'
if [ -z "$3" ]; then
    acc_or_perf="performance"
else
    acc_or_perf="$3"
fi

# 'false' or 'true', enable this config to profile compiled module
if [ -z "$4" ]; then
    enable_inductor_profile="false"
else
    enable_inductor_profile=$4
fi

# 'bfloat16', 'float16', 'float32' or 'amp'
if [ -z "$5" ]; then
    precision="float32"
else
    precision=$5
fi



batch_size=1
found="false"

while IFS= read -r line; do
    clean_line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    cur_model_name=$(echo "$clean_line" | awk '{print $1}')
    cur_batch_size=$(echo "$clean_line" | awk '{print $2}')


    if [ "$model_name" == "$cur_model_name" ]; then
        found="true"
        batch_size=$cur_batch_size
        echo "model: $model_name, batch_size: $batch_size"
        break
    fi
done < "$MODEL_LIST"

if $found; then
    echo "- strat run $model_name -"
    echo "- profile trace path $trace_path -"
    if [[ $enable_inductor_profile == "true" && $acc_or_perf == "performance" ]]; then
        echo "- enable torchinductor GPU profiling -"
        TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 TORCH_COMPILE_DEBUG=1 \
        TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 TORCHINDUCTOR_BENCHMARK_KERNEL=1 python -u ../dynamo/timm_models.py --$acc_or_perf  -dcuda \
                            --output=$current_path/dump_graphs/torchbench_fp32_training_perf.csv \
                            --$test_mode --inductor --no-skip --dashboard \
                            --only $model_name --batch-size $batch_size --$precision \
                            --disable-cudagraphs \
                            2>&1 | tee $current_path/${model_name}_bs${batch_size}_${precision}_${test_mode}.txt_py
    else
        python -u ../dynamo/timm_models.py --$acc_or_perf  -dcuda \
                            --output=$current_path/dump_graphs/torchbench_fp32_training_perf.csv \
                            --$test_mode --inductor --no-skip --dashboard \
                            --only $model_name --batch-size $batch_size --$precision --disable-cudagraphs \
                            2>&1 | tee $current_path/${model_name}_bs${batch_size}_${precision}_${test_mode}.txt_py
    fi
# # TODO: should read and run all model from $MODEL_LIST
# elif [[ "$model_name" == "run_all" ]]; then
#     echo "- start run all network in list -"
#     for i in "${!nets_list[@]}"; do
#         model_name=${nets_list[$i]}
#         batch_size=${batchsize_list[$i]}

#         echo "- strat run $model_name - "
#         python ./dynamo/timm_models.py --$acc_or_perf  -dcuda \
#                             --output=$current_path/dump_graphs/torchbench_fp32_training_perf.csv \
#                             --$test_mode --inductor --no-skip --dashboard \
#                             -k $model_name --batch-size $batch_size --$precision\
#     done
else
    echo "- illeage network name! retry please -"
fi

if [[ $enable_inductor_profile == "true" && $acc_or_perf == "performance" ]]; then
    echo " " >> tot_mx_log_torch
    echo "model: ${model_name}, batch-size: ${batch_size}, precision: ${precision}, mode: ${test_mode}" >> tot_mx_log_torch
    # extract module file path
    readarray -t module_paths < <(sed -n 's/.*Compiled module path: \(.*\)/\1/p' "$current_path/${model_name}_bs${batch_size}_${precision}_${test_mode}.txt_py")

    echo "num of module paths: ${#module_paths[@]}"
    if [[ $test_mode == "training" && ${#module_paths[@]} -lt 2 ]]; then
        echo "Error: should get at least 2 module paths, but get ${#module_paths[@]} instead" >&2
        exit 1
    elif [[ $test_mode == "inference" && ${#module_paths[@]} -ne 1 ]]; then
        echo "Error: should get 1 module path, but get ${#module_paths[@]} instead" >&2
        exit 1
    fi

    log_paths=()
    if [ $test_mode == "training" ]; then
        # training mode might have various types and number of compiled module files
        # need to confirm module type by from module

        for ((i=0; i<${#module_paths[@]}; i++)); do
            log_paths+=("$current_path/${model_name}_bs${batch_size}_${precision}_${test_mode}_${i}.log")
        done

        if [ ${#module_paths[@]} -ne ${#log_paths[@]} ]; then
            echo "module paths should equal to log paths!"
        fi


    else
        inference_module_path="${module_paths[0]}"
        echo "inference module path: $inference_module_path"
        inference_log_path="$current_path/${model_name}_bs${batch_size}_${precision}_${test_mode}_inference.log"
        log_paths+=($inference_log_path)
        echo "inference_log_path path: ${log_paths[0]}"
    fi

    if [  ! -f "${module_paths[0]}" ]; then
        echo "Compiled module files not found!"
        echo "Run 'rm -r /tmp/torchinductor_{user_name}' to remove compiled moduled cache"
        echo " "
        echo " "
        echo " "
        exit 1
    fi

    num_of_modules=${#log_paths[@]}

    for ((i=0; i<$num_of_modules; i++)); do
        # # use 'awk' to enable tf32 config after "import torch"
        # awk '
        # /import torch/ {
        # print
        # print "torch.backends.cuda.matmul.allow_tf32 = True"
        # print "torch.backends.cudnn.allow_tf32 = True"
        # next
        # }
        # 1' "${module_paths[$i]}" > temps && mv temps "${module_paths[$i]}"

        # echo "enable tf32 successfully!"

        python ${module_paths[$i]} -p > ${log_paths[$i]}

        echo "profile_log_$i path: ${log_paths[$i]}"
        # calculate kernel time
        awk '
        /  == triton_/ {
            if (!found_kernel) {
                found_kernel = 1
            }
            next
        }
        found_kernel && /^triton_poi_fused/ {
            count++
        }
        END {
            print "triton_poi_fused_ appears ", count, "times." >> "tot_mx_log_torch"
        }
        ' "${log_paths[$i]}"

        awk '
        /  == triton_/ {
            if (!found_kernel) {
                found_kernel = 1
            }
            next
        }
        found_kernel && /^triton_red_fused_/ {
            count++
        }
        END {
            print "triton_red_fused_ appears ", count, "times." >> "tot_mx_log_torch"
        }
        ' "${log_paths[$i]}"

        awk '
        /  == triton_/ {
            if (!found_kernel) {
                found_kernel = 1
            }
            next
        }
        found_kernel && /^triton_per_fused_/ {
            count++
        }
        END {
            print "triton_per_fused_ appears ", count, "times." >> "tot_mx_log_torch"
        }
        ' "${log_paths[$i]}"

        # model may not have all 3 types of kernel, so need to calcuate wiht dynamic array
        awk '
        /Total/ {
        for (i=2; i<=NF; i++) {
            if ($i ~ /^[0-9]+\.[0-9]+$/) {
                #print $i
                f[++count] = $i
                break
            }
            }
        }
        / unknown category kernels/{
            if (!found_other){
                found_other = 1
            }
            next
        }
        END {
        sum = 0
            if (found_other){
                for (i=1; i<count-1; i++) {
                    sum += f[i]
                }
                print "triton kernel total time(ms): " sum  >> "tot_mx_log_torch"
                print "other kernel total time(ms): " f[count-1]  >> "tot_mx_log_torch"
            }
            else {
                for (i=1; i<count; i++) {
                    sum += f[i]
                }
                print "tritonkernel total time (ms): " sum  >> "tot_mx_log_torch"
                print "other kernel total time(ms): 0" >> "tot_mx_log_torch"
            }
        }
        ' "${log_paths[$i]}"
        echo " "
        echo " "
        echo " " >> "tot_mx_log_torch"
    done
    echo "check summary at './tot_mx_log_torch'"
fi