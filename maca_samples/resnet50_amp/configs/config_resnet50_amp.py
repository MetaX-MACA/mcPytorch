import os
golden_data_path = "/netapp/pytorch/golden/amp/resnet50"
resnet50_model_path = "/netapp/pytorch/golden/resnet50_new/model/"
resnet50_model_name = 'resnet50-0676ba61.pth'
error_eps = 1e-3
loss_eps = 0.035

GOLDEN_PATH = os.getenv("PYTORCH_TEST_GOLDEN_PATH")
if GOLDEN_PATH and os.path.exists(GOLDEN_PATH):
    resnet50_model_path = os.path.join(GOLDEN_PATH, "resnet50_new", "model", "")
    golden_data_path = os.path.join(GOLDEN_PATH, "amp", "resnet50", "")
