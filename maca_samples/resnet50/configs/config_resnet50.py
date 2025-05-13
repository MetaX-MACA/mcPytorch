import os
error_eps = 1e-2
download_model = False
create_input_tensor = False
create_label_tensor = False
create_golden_data = False
create_golden_type = 'all_cpu'  # 'all_cpu', 'all_gpu'
check_data = True
conv_to_cpu = False
is_backward = True
is_check_from_hook = True
is_check_output = True


resnet50_model_download_path = './model/'
resnet50_model_path = '/netapp/pytorch/golden/resnet50_new/model/'
resnet50_model_name = 'resnet50-0676ba61.pth'
write_input_path = './'
read_input_path = '/netapp/pytorch/golden/resnet50_new/'
write_golden_path = './'
read_golden_path = '/netapp/pytorch/golden/resnet50_new/'
result_path = './'

GOLDEN_PATH = os.getenv("PYTORCH_TEST_GOLDEN_PATH")
if GOLDEN_PATH and os.path.exists(GOLDEN_PATH):
    resnet50_model_path = os.path.join(GOLDEN_PATH, "resnet50_new", "model", "")
    read_input_path = os.path.join(GOLDEN_PATH, "resnet50_new", "")
    read_golden_path = os.path.join(GOLDEN_PATH, "resnet50_new", "")
