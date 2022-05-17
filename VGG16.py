# -*- coding: utf-8 -*-
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
import torchvision
import torch
model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


# def Backbone_ResNet50_in3():
#     net = resnet50(pretrained=True)
#     div_2 = nn.Sequential(*list(net.children())[:3])
#     div_4 = nn.Sequential(*list(net.children())[3:5])
#     div_8 = net.layer2
#     div_16 = net.layer3
#     div_32 = net.layer4
#
#     return div_2, div_4, div_8, div_16, div_32


def Backbone_VGG16_in3():
    net = vgg16_bn(pretrained=False, progress=True)
    # net.load_state_dict(torch.load('vgg16.pth'))

    div_1 = nn.Sequential(*list(net.children())[0][0:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16

class Up(nn.Module):
    def __init__(self, ic, oc, scale_factor=2):
        super(Up, self).__init__()
        self.conv_fg = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(oc, oc, kernel_size=3, padding=1), nn.ReLU()
        )
        self.scale_factor = scale_factor

    def interp(self, x, scale_factor):
        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        return x
    def forward(self, input,top):

        x = self.conv_fg(input)
        x = self.interp(x,scale_factor=self.scale_factor)
        x = torch.cat([x, top], dim=1)

        return x

class Catconv(nn.Module):
    def __init__(self,ic, oc, mc):
        super(Catconv, self).__init__()
        self.conv = nn.Conv2d(ic,oc,3,padding=1)
        self.Relu = nn.ReLU()
        self.downconv = nn.Conv2d(oc + mc, oc, kernel_size=1)

    def interp(self, x, scale_factor):
        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        return x

    def forward(self, input, pre):

        x = self.Relu(self.conv(input))
        up = self.interp(pre, scale_factor=2)
        cat = torch.cat([x,up],dim=1)
        out = self.Relu(self.downconv(cat))

        return  out+input

class Catconvlast(nn.Module):
    def __init__(self,ic, oc, mc):
        super(Catconvlast, self).__init__()
        self.conv = nn.Conv2d(ic,oc,3,padding=1)
        self.Relu = nn.ReLU()
        self.downconv = nn.Conv2d(oc + mc, oc, kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor([0.1]))

    def interp(self, x, scale_factor):
        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        return x

    def forward(self, input, pre,start):

        x = self.Relu(self.conv(input))
        up = self.interp(pre, scale_factor=2)
        cat = torch.cat([x,up],dim=1)
        out = self.Relu(self.downconv(cat))
        output = self.delta*out+start

        return  output

class Resconv(nn.Module):
    def __init__(self, ic, ot):
        super(Resconv, self).__init__()
        self.conv = nn.Conv2d(ic,ot,kernel_size=3,padding=1)

    def forward(self,x):
        out = self.conv(x)

        return out+x




if __name__ == '__main__':
    div_1, div_2, div_4, div_8, div_16 = Backbone_VGG16_in3()
    print(div_1)
    print(div_2)
    print(div_4)
    print(div_8)
    print(div_16)