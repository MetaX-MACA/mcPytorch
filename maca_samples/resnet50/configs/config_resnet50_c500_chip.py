error_eps = 1e-2
download_model = False
create_input_tensor = True
create_label_tensor = True
create_golden_data = True
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
read_input_path = './'
write_golden_path = './'
read_golden_path = './'
result_path = './'
