import torch
import argparse
import dlrm_module
from itertools import repeat


if __name__ == "__main__":
    # init batch size
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, help="batch size")
    parser.add_argument("--null_hardware", action="store_true", default=False, help="null_hardware for driver")
    parser.add_argument("--loop_num", default=1, help="loop num")
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    dlrm_module.set_batch_size(batch_size)

    # init input tensor and target tensor
    input_dense_cpu, input_sparse_offset_cpu, input_sparse_cpu, target_cpu = dlrm_module.create_input_data(batch_size)
    input_dense_gpu = input_dense_cpu.cuda()
    input_sparse_offset_gpu = input_sparse_offset_cpu.cuda()
    input_sparse_gpu = [element.cuda() for element in input_sparse_cpu]
    target_gpu = target_cpu.cuda()

    # create resutl file
    if args.null_hardware == False:
        dlrm_module.create_result_file()

    # create golden data output from cpu
    dlrm_module.set_test_mode("golden_mode", args.null_hardware)
    dlrm_cpu = dlrm_module.DLRM_Net()
    output_cpu = dlrm_cpu(input_dense_cpu, input_sparse_offset_cpu, input_sparse_cpu)
    loss_cpu = dlrm_cpu.loss_fn(output_cpu, target_cpu)
    loss_cpu.backward()

    # create test data output from maca/cuda
    dlrm_module.set_test_mode("test_mode", args.null_hardware)
    dlrm_gpu = dlrm_module.DLRM_Net().cuda()
    if int(args.loop_num) != -1:
        loops = range(int(args.loop_num))
    else:
        loops = repeat(None)
    idx = 0
    for _ in loops:
        print("loop: ", idx)
        output_gpu = dlrm_gpu(input_dense_gpu, input_sparse_offset_gpu, input_sparse_gpu)
        loss_gpu = dlrm_gpu.loss_fn(output_gpu, target_gpu)
        loss_gpu.backward()
        idx += 1

    # print result
    if args.null_hardware == False:
        print("output_cpu:", output_cpu)
        print("loss_cpu:", loss_cpu)
        print("output_gpu:", output_gpu)
        print("loss_gpu:", loss_gpu)
        print("DLRM test checkout finish!")

    ret = True
    if ret:
        print("Passed: {}".format(__file__))
        exit(0)
    else:
        print("Failed: {}".format(__file__))
        exit(1)
