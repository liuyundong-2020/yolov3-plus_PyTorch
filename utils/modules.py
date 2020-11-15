import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os



def get_device(gpu_ind):
    if torch.cuda.is_available():
        print('Let us use GPU.')
        cudnn.benchmark = True
        if torch.cuda.device_count() == 1:
            device = torch.device('cuda')
        else:
            device = torch.device('cuda:%d' % gpu_ind)
    else:
        print('Come on !! No GPU ?? Who gives you the courage to study Deep Learning ?')
        device = torch.device('cpu')

    return device


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


# Copy from yolov5
class Focus(nn.Module):
    """
        Focus module proposed by yolov5.
    """
    # Focus wh information into c-space
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=None, groups=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv2d(in_channels=in_ch, out_channels=out_ch, ksize=ksize, stride=stride, leakyReLU=act)

    def forward(self, x):  # x(B, C, H, W) -> y(B, 4C, H/2, W/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# Mish https://github.com/digantamisra98/Mish
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()