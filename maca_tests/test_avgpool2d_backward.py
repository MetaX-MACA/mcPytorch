import torch

def test_avgpool2d_backward():
    shape_list=[ # shape in the model
                [(128, 384, 14, 14), (1, 1), (1, 1), (0, 0)], 
                [(128, 384, 28, 28), (2, 2), (2, 2), (0, 0)],
                [(128, 384, 56, 56), (4, 4), (4, 4), (0, 0)],
                # shape not in the model, but pattern supported in opt
                [(32, 33, 14, 14), (2, 2), (2, 2), (0, 0)],
                [(32, 33, 28, 28), (4, 2), (4, 2), (0, 0)],
                [(32, 33, 7, 7), (7, 7), (7, 7), (0, 0)],
                [(1, 33, 64, 64), (16, 16), (16, 16), (0, 0)],
                [(1, 33, 45, 45), (9, 9), (9, 9), (0, 0)],
                [(16, 33, 72, 36), (4, 2), (4, 2), (0, 0)],
                # pattern not supported in opt
                [(32, 64, 5, 5), (2, 2), (2, 2), (0, 0)],
                [(32, 33, 14, 14), (2, 4), (2, 4), (0, 0)],
                [(32, 33, 7, 7), (3, 3), (3, 3), (0, 0)],
                [(32, 33, 7, 7), (3, 3), (3, 3), (1, 1)],
                [(32, 33, 49, 49), (5, 5), (2, 2), (1, 1)],
                [(32, 33, 128, 128), (16, 8), (8, 4), (1, 1)]
                ]
    
    for shape, kernel, stride, pad in shape_list:
        m = torch.nn.AvgPool2d(kernel, stride=stride, padding=pad)

        input_d = torch.randn(shape, dtype=torch.float).cuda()
        input_c = input_d.cpu()
        input_d.requires_grad = True
        input_c.requires_grad = True

        output_d = m(input_d)
        output_c = m(input_c)

        grad_d = torch.randn(output_d.shape).cuda()
        grad_c = grad_d.cpu()

        output_d.backward(gradient=grad_d)
        output_c.backward(gradient=grad_c)

        input_grad_d = input_d.grad.cpu()
        input_grad_c = input_c.grad

        if not torch.allclose(input_grad_d, input_grad_c):
            print("shape=", shape, " kernel=", kernel, " stride=", stride, " pad=", pad, "avgpool Error!")
            # print(input_grad_d)
            # print(input_grad_c)
            exit(1)

test_avgpool2d_backward()
