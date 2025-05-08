PROFILING
===================

Provide single op test scripts in Resnet50 and Bert base network.

Resnet50
--------------------------
Usage, e.g.:

    python test_resnet50_ops.py --op_type batchnorm --id 0  --only_run --only_fwd --batch_size 1

- op_type: ops of different types
- id: ops of different input sizes
- only_run: used for profile, not check accuracy, only run fwd (and bww + bwd)
- only_fwd: only run fwd, not run backward (bwd + bww). Default is fwd + bww + bwd.
- batch_size: Default is 1. Only 1,2,4 is verified well using fast model env and 1 is verified in full model env.

Test cases are provided as:

- python test_resnet50_ops.py --op_type batchnorm --id [0-11]  --only_run
- python test_resnet50_ops.py --op_type avgpool --id [0] --only_run
- python test_resnet50_ops.py --op_type maxpool --id [0] --only_run
- python test_resnet50_ops.py --op_type relu --id [0-11] --only_run
- python test_resnet50_ops.py --op_type linear --id [0]  --only_run
- python test_resnet50_ops.py --op_type add --id [0-3]  --only_run
- python test_resnet50_ops.py --op_type conv2d --id [0-22] --only_run

Bert
--------------------------
