import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..registry import HEADS
import numpy as np

class DarknetConv2D_BN_Leaky(nn.Module):
    def __init__(self, numIn, numOut, ksize, stride=1, padding=1):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        self.conv1 = nn.Conv2d(numIn, numOut, ksize, stride, padding, bias=False)  # regularizer': l2(5e-4)
        self.bn1 = nn.BatchNorm2d(numOut)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyReLU(x)
        return x

class LastLayer(nn.Module):
    def __init__(self, numIn, numOut, numOut2):
        super(LastLayer, self).__init__()
        self.dark_conv1 = DarknetConv2D_BN_Leaky(numIn, numOut, ksize=1, stride=1, padding=0)
        self.dark_conv2 = DarknetConv2D_BN_Leaky(numOut, numOut * 2, ksize=3, stride=1, padding=1)
        self.dark_conv3 = DarknetConv2D_BN_Leaky(numOut * 2, numOut, ksize=1, stride=1, padding=0)
        self.dark_conv4 = DarknetConv2D_BN_Leaky(numOut, numOut * 2, ksize=3, stride=1, padding=1)
        self.dark_conv5 = DarknetConv2D_BN_Leaky(numOut * 2, numOut, ksize=1, stride=1, padding=0)

        self.dark_conv6 = DarknetConv2D_BN_Leaky(numOut, numOut * 2, ksize=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(numOut * 2, numOut2, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.dark_conv1(x)
        x = self.dark_conv2(x)
        x = self.dark_conv3(x)
        x = self.dark_conv4(x)
        x = self.dark_conv5(x)

        y = self.dark_conv6(x)
        y = self.conv7(y)
        return x, y

class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

@HEADS.register_module
class YoloHead(nn.Module):
    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=[256, 512, 1024],
                 anchors=[[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]],
                 num_classes=80,
                 input_size=416,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(YoloHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False

        self.input_size = input_size
        self.grid_size = [int(input_size/8), int(input_size/16), int(input_size/32)]
        self.anchors = anchors
        in_channels = self.in_channels
        self.head_layers = []


        for i in range(len(in_channels)):
            numIn = int(in_channels[i])
            # if numIn != 1024:
            #     numIn = numIn  + int(numIn / 2)
            numOut = int(numIn / 2)
            numOut2 = 3 * (self.num_classes - 1 + 5)

            if numIn != 1024:
                up = nn.Sequential(DarknetConv2D_BN_Leaky(numIn, numOut, ksize=1, stride=1, padding=0),
                                   Upsample(scale_factor=2))
                last = LastLayer(numIn + numOut, numOut, numOut2)
                up_name = 'up{}'.format(i + 1)
                self.add_module(up_name, up)
                last_name = 'last{}'.format(i + 1)
                self.add_module(last_name, last)
                self.head_layers.append([up_name, last_name])
            else:
                last = LastLayer(numIn, numOut, numOut2)
                last_name = 'last{}'.format(i + 1)
                self.add_module(last_name, last)
                self.head_layers.append(last_name)

    def init_weights(self):
        pass

    @auto_fp16()
    def forward(self, input):
        output = []
        x = input[-1]
        for i in range(len(self.head_layers) - 1, -1, -1):
            layers_name = self.head_layers[i]
            if isinstance(layers_name, str):
                layer = getattr(self, layers_name)
                if isinstance(layer, LastLayer):
                    x, y = layer(x)
                    output.append(y)
            elif isinstance(layers_name, list):
                for layer_name in layers_name:
                    layer = getattr(self, layer_name)
                    if isinstance(layer, nn.Sequential):
                        x = layer(x)
                        x = torch.cat((x, input[i]), 1)
                    elif isinstance(layer, LastLayer):
                        x, y = layer(x)
                        output.append(y)
                    else:
                        print("error")
        output = output[::-1]
        return (output, )

    def wh_iou(self, box1, box2):
        # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
        box2 = box2.t()

        # w, h = box1
        w1, h1 = box1[0], box1[1]
        w2, h2 = box2[0], box2[1]

        # Intersection area
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)

        # Union Area
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

        return inter_area / union_area  # iou

    def build_targets(self, gt_bboxes, gt_labels):
        # targets = [image, class, x, y, w, h]
        iou_thres = 0.3 # hyperparameter

        targets = []
        for i, (gt_bbox, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
            assert torch.min(gt_label) > 0
            size = gt_label.size()[0]
            idx = torch.full((size, 1), i).float()

            if gt_label.is_cuda:
                idx = idx.cuda()
            gt_label = (gt_label - 1).view(-1, 1).float()
            g_target = torch.cat((idx, gt_label, gt_bbox), 1)
            targets.append(g_target)
        targets = torch.cat(targets, 0)

        nt = len(targets)
        txy, twh, tcls, tbox, indices, anchor_vec = [], [], [], [], [], []
        for i in range(len(self.grid_size)):
            stride = self.input_size / self.grid_size[i]
            an_vector = torch.FloatTensor(self.anchors[i * 3: (i + 1) * 3]).to(targets.device) / stride
            # iou of targets-anchors
            t, a = targets, []
            gwh = (targets[:, 4:6] - targets[:, 2:4] + 1) / stride
            if nt:
                iou = [self.wh_iou(x, gwh) for x in an_vector]
                iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor

                reject = True
                if reject:
                    j = iou > iou_thres
                    t, a, gwh = targets[j], a[j], gwh[j]

            b, c = t[:, :2].long().t()  # target image, class
            gxy = (t[:, 2:4] + t[:, 4:6]) / 2 / stride  # grid x_center, y_center
            gi, gj = gxy.long().t()  # grid x, y indices
            indices.append((b, a, gj, gi))

            gxy -= gxy.floor()
            txy.append(gxy)
            tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
            anchor_vec.append(an_vector[a])
            twh.append(torch.log(gwh / an_vector[a]))  # wh yolo method
            tcls.append(c)

        return txy, twh, tcls, tbox, indices, anchor_vec


    @force_fp32(apply_to=('output', ))
    def loss(self,
             output,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        #  output, gt_bboxes, gt_labels, img_metas, self.train_cfg
        losses = dict()
        assert len(output) == 3

        bs = len(gt_bboxes)

        ft = torch.cuda.FloatTensor
        lxy, lwh, lcls, lconf = ft([0]), ft([0]), ft([0]), ft([0])
        txy, twh, tcls, tbox, indices, anchor_vec = self.build_targets(gt_bboxes, gt_labels)

        MSE = nn.MSELoss()
        CE = nn.CrossEntropyLoss()  # (weight=model.class_weights)
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([79.0]))
        BCEconf = nn.BCEWithLogitsLoss(pos_weight=ft([3.53]))

        k = bs
        for i, p in enumerate(output):  # layer i predictions, i
            ss = len(anchor_vec)
            p = p.view(bs, len(anchor_vec), self.num_classes - 1 + 5, int(self.grid_size[i]), int(self.grid_size[i])).permute(0, 1, 3, 4, 2).contiguous()
            bs = p.shape[0]  # batch size

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tconf = torch.zeros_like(p[..., 0])  # conf

            if len(b):
                pi = p[b, a, gj, gi]  # predictions closest to anchors
                tconf[b, a, gj, gi] = 1.0  # conf

                lxy += (k * 0.2) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
                lwh += (k * 0.1) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss

                # lcls += (k * h['cls']) * BCEcls(pi[..., 5:], tclsm)  # cls loss (BCE)
                lcls += (k * 0.035) * CE(pi[..., 5:], tcls[i])  # cls loss (CE)
            lconf += (k * 1.61) * BCEconf(p[..., 4], tconf)  # obj_conf loss
        # loss = lxy + lwh + lconf + lcls
        losses["loss_xy"] = lxy
        losses["loss_wh"] = lwh
        losses["loss_conf"] = lconf
        losses["loss_cls"] = lcls
        return losses

    @force_fp32(apply_to=('output', ))
    def get_bboxes(self, output, img_metas, cfg,
                   rescale=False):
        num_levels = len(output)

        result_list = []
        bs = len(img_metas)
        io_list = []
        for i, p in enumerate(output):
            nx, ny = (self.grid_size[i], self.grid_size[i])  # x and y grid size
            stride = self.input_size / nx

            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).to(p.device).float().view((1, 1, ny, nx, 2))

            anchor_vec = torch.FloatTensor(self.anchors[i * 3: (i + 1) * 3]).to(p.device) / stride
            anchor_wh = anchor_vec.view(1, 3, 1, 1, 2).to(p.device)

            p = p.view(bs, 3, self.num_classes - 1 + 5, self.grid_size[i], self.grid_size[i]).permute(0, 1, 3, 4, 2).contiguous()
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
            io[..., :4] *= stride
            io = io.view(bs, -1, 5 + self.num_classes - 1)
            io_list.append(io)
        res_io = torch.cat(io_list, 1)
        dets = self.non_max_suppression(res_io, conf_thres=0.1, nms_thres=0.5)
        result = []
        # *xyxy, conf, cls_conf, cls
        for i, det in enumerate(dets):
            if det is None:
                result.append((torch.Tensor([]), torch.Tensor([])))
            else:
                det[:, :4] = self.scale_coords(img_metas[i]['img_shape'][:2], det[:, :4], img_metas[i]['ori_shape'][:2]).round()
                result.append((det[:, :5], det[:, 6]))
        return result

    def scale_coords(self, img1_shape, coords, img0_shape):
        # Rescale coords1 (xyxy) from img1_shape to img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
        coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
        coords[:, :4] /= gain
        coords[:, :4] = coords[:, :4].clamp(min=0)
        return coords

    def xywh2xyxy(self, x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def create_grids(self, img_size=416, ng=(13, 13), device='cpu'):
        nx, ny = ng  # x and y grid size
        self.img_size = img_size
        self.stride = img_size / max(ng)
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, ny, nx, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(device) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)
        self.ng = torch.Tensor(ng).to(device)
        self.nx = nx
        self.ny = ny

    def non_max_suppression(self, prediction, conf_thres=0.1, nms_thres=0.5):
        min_wh = 2
        output = [None] * len(prediction)
        for image_i, pred in enumerate(prediction):

            class_conf, class_pred = pred[:, 5:].max(1)
            pred[:, 4] *= class_conf

            i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
            pred = pred[i]

            if len(pred) == 0:
                continue
            class_conf = class_conf[i]
            class_pred = class_pred[i].unsqueeze(1).float()
            pred[:, :4] = self.xywh2xyxy(pred[:, :4])
            pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)
            pred = pred[(-pred[:, 4]).argsort()]
            det_max = []
            nms_style = 'MERGE'
            for c in pred[:, -1].unique():
                dc = pred[pred[:, -1] == c]  # select class c
                n = len(dc)
                if n == 1:
                    det_max.append(dc)
                    continue
                elif n > 300:
                    dc = dc[:300]

                if nms_style == 'OR':
                    while dc.shape[0]:
                        det_max.append(dc[:1])
                        if len(dc) == 1:
                            break
                        iou = self.bbox_iou(dc[0], dc[1:])
                        dc = dc[1:][iou < nms_thres]

                elif nms_style == 'AND':
                    while len(dc) > 1:
                        iou = self.bbox_iou(dc[0], dc[1:])
                        if iou.max() > 0.5:
                            det_max.append(dc[:1])
                        dc = dc[1:][iou < nms_thres]

                elif nms_style == 'MERGE':
                    while len(dc):
                        if len(dc) == 1:
                            det_max.append(dc)
                            break
                        i = self.bbox_iou(dc[0], dc) > nms_thres
                        weights = dc[i, 4:5]
                        dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                        det_max.append(dc[:1])
                        dc = dc[i == 0]

                elif nms_style == 'SOFT':
                    sigma = 0.5
                    while len(dc):
                        if len(dc) == 1:
                            det_max.append(dc)
                            break
                        det_max.append(dc[:1])
                        iou = self.bbox_iou(dc[0], dc[1:])
                        dc = dc[1:]
                        dc[:, 4] *= torch.exp(-iou ** 2 / sigma)
                        dc = dc[dc[:, 4] > nms_thres]
            if len(det_max):
                det_max = torch.cat(det_max)
                output[image_i] = det_max[(-det_max[:, 4]).argsort()]
        return output


    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.t()

        # Get the coordinates of bounding boxes
        if x1y1x2y2:
            # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            # x, y, w, h = box1
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                     (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                     (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

        iou = inter_area / union_area  # iou
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
            c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
            c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
        return iou
