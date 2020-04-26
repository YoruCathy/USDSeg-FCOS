import torch
import torch.nn as nn
from mmcv.cnn import normal_init


@HEADS.register_module
class CornerNetHead(nn.Module):
    """
    Implementation of CornerNet based on mmdetection framwork.
    """
    def __init__(self,num_classes,in_channels,):
        super(CornerNetHead,self).__init__()

        self.num_classes=num_classes
        self.cls_out_channels = num_classes-1
        self.in_channels=in_channels
        

