import torch
import torchvision
import onnx
import traceback

def export_onnx_file():
    dynamic_axes = [{
            "actual_input_1":{0:"batch_size", 2:"in_width", 3:"in_height"},
            "output1":{0:"batch_size", 2:"out_width", 3:"out_height"}
        },
        None]
    for dynamic_axe in dynamic_axes:
        onnx_file_name = "resnet18.onnx"
        dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
        model = torchvision.models.resnet18().cuda()
        input_names = ["actual_input_1"]
        output_names = ["output1"]
        
        torch.onnx.export(model, dummy_input, onnx_file_name, verbose=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axe)
        model = onnx.load(onnx_file_name)
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))

if __name__ == '__main__':
    try:
        export_onnx_file()
    except:
        traceback.print_exc()
        exit(1)
    exit(0)