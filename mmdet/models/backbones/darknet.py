import logging
import torch.nn as nn
from mmdet.models.utils import build_conv_layer, build_norm_layer
from mmcv.cnn import constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES

def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):

    layers = []

    layers.append(nn.Conv2d(inplanes, planes, kernel_size=3,
                                        stride=2, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(planes))
    layers.append(nn.LeakyReLU(0.1))

    inplanes = planes * block.expansion
    for i in range(0, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, int(planes/2), postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            int(planes/2),
            1,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = build_conv_layer(
            conv_cfg, int(planes/2), planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu2= nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


@BACKBONES.register_module
class DarkNet(nn.Module):
    arch_settings = {
        21: (BasicBlock, (1, 1, 2, 2, 1)),
        53: (BasicBlock, (1, 2, 8, 8, 4)),
    }

    def __init__(self,
                 input_size,
                 depth,
                 strides=(1, 1, 1, 1, 1),
                 dilations=(1, 1, 1, 1, 1),
                 out_indices = (2, 3, 4),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 style = 'pytorch',
                 ):
        super(DarkNet, self).__init__()
        self.input_size = input_size
        self.out_indices = out_indices
        if self.input_size % 32 != 0:
            raise ValueError('Input size must be a multiple of 32')

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.style = style

        self.block, stage_blocks = self.arch_settings[depth]
        self.stags_blocks = stage_blocks

        self.inplanes = 32
        self._make_stem_layer()
        self.res_layers = []

        for i, num_blocks in enumerate(self.stags_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i

            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style
                )
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, self.inplanes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = nn.LeakyReLU(0.1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')