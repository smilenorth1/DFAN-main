import torch
import torch.nn as nn

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,bit_depth):
        input = torch.clamp(input, 0, 1)
        max_value=2**bit_depth-1.0
        output = (input * max_value).round() / max_value

        # input = torch.clamp(input, 0, 65535)
        # input=input/65535.0
        # max_value=2**bit_depth-1.0
        # output = (input * max_value).floor() / max_value
        # output=output*65535.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output,None #grad_output is the gradient passed to this layer. There two return-value because there are to inputs in forward. the second 
                                #return value is None because the second input is a constant

class Quantization(nn.Module):
    def __init__(self):
        
        super(Quantization, self).__init__()
        # print("!!!!!!!!!!!!!!!!!!!!")
    def forward(self, input, bit_depth):
        # print("!!!!!!!!!!!!!!!!!!!!")
        return Quant.apply(input,bit_depth)
