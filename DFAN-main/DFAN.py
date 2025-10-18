import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import ops
import SwinT

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


class MRB(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(MRB, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_feature, out_channels=2 * in_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2_1 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=1, stride=1, padding=0, bias=True, groups=2 * in_feature)
        self.Conv2_2 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=3, stride=1, padding=1, bias=True, groups=2 * in_feature)
        self.Conv2_3 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=5, stride=1, padding=2, bias=True, groups=2 * in_feature)
        self.Conv2_4 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=7, stride=1, padding=3, bias=True, groups=2 * in_feature)
        self.Conv3 = nn.Conv2d(in_channels=2 * in_feature, out_channels=in_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv_end = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.LRelu = nn.LeakyReLU()
    def forward(self, x):
        out1 = self.LRelu(self.Conv1(x))
        out21 = self.LRelu(self.Conv2_1(out1))
        out22 = self.LRelu(self.Conv2_2(out1))
        out23 = self.LRelu(self.Conv2_3(out1))
        out24 = self.LRelu(self.Conv2_4(out1))
        out3 = torch.add(out1, (out21 + out22 + out23 + out24))
        out4 = self.LRelu(self.Conv3(out3))
        out5 = torch.add(x, out4)
        out6 = self.LRelu(self.Conv_end(out5))
        return out6


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class HBCT(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(HBCT, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.c1_r = conv_layer(2*in_channels, self.rc, 3)

        self.esa = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        # self.sparatt = Spartial_Attention.Spartial_Attention()
        self.mrb = MRB(in_channels, in_channels)
        self.swinT = SwinT.SwinT()

    def forward(self, input):
        input = self.esa2(input)
        input1 = self.mrb(input)
        input2 = self.swinT(input)
        out_fused = self.esa(self.c1_r(torch.concat([input1, input2], dim=1)))
        return out_fused


class Cascade(nn.Module):
    def __init__(self, ):
        super(Cascade, self).__init__()
        self.conv1 = conv_layer(50, 50, kernel_size=1)
        self.conv3 = conv_layer(50, 50, kernel_size=3)
        self.conv5 = conv_layer(50, 50, kernel_size=5)
        self.c = conv_block(50 * 4, 50, kernel_size=1, act_type='lrelu')

    def forward(self, x):
        conv5 = self.conv5(x)
        extra = x+conv5
        conv3 = self.conv3(extra)
        extra = x + conv3
        conv1 = self.conv1(extra)
        cat = torch.cat([conv5, conv3, conv1, x], dim=1)
        input = self.c(cat)
        return input


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class DAFN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=12, upscale=4):
        super(DAFN, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = HBCT(in_channels=nf)
        self.B2 = HBCT(in_channels=nf)
        self.B3 = HBCT(in_channels=nf)
        self.B4 = HBCT(in_channels=nf)
        self.B5 = HBCT(in_channels=nf)
        self.B6 = HBCT(in_channels=nf)
        self.B7 = HBCT(in_channels=nf)
        self.B8 = HBCT(in_channels=nf)
        self.B9 = HBCT(in_channels=nf)
        self.B10 = HBCT(in_channels=nf)
        self.B11 = HBCT(in_channels=nf)
        self.B12 = HBCT(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        # self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        self.upsampler = conv_layer(nf, in_nc, kernel_size=3)
        self.scale_idx = 0
    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        out_B9 = self.B9(out_B8)
        out_B10 = self.B10(out_B9)
        out_B11 = self.B11(out_B10)
        out_B12 = self.B12(out_B11)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11, out_B12], dim=1))
        # out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_B + out_fea)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # x = torch.randn(1, 3, 640, 360)
    x = torch.randn(1, 3, 320, 180)

    model = DAFN(in_nc=3, nf=36, num_modules=12, upscale=1)
    # print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)