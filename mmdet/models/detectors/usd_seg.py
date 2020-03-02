from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn

from mmdet.core import bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector

import numpy as np


@DETECTORS.register_module
class USDSeg(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 bases_path=None,
                 method='None'):
        super(USDSeg, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)

        if bases_path is None:
            raise RuntimeWarning('bases_path not defined!')
        else:
            self.bases = torch.tensor(np.load(bases_path)).pin_memory()
            self.bases_copied = False
            self.num_bases = len(self.bases)

        if method not in ['var', 'cosine']:
            raise NotImplementedError('%s not supported.' % method)
        self.method = method

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_coefs,
                      gt_bboxes_ignore=None,
                      ):

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_coefs, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_bboxes_ignore=gt_bboxes_ignore,
        )
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        if not self.bases_copied:
            self.bases = self.bases.to(img.device).float()
            self.bases_copied = True

        if self.method == 'var':
            from mmdet.datasets.pipelines.coefs import x_mean_32, sqrt_var_32
            x_mean_32 = x_mean_32.to(bbox_list[0][2].device).float()
            sqrt_var_32 = sqrt_var_32.to(bbox_list[0][2].device).float()

            results = [
                bbox_mask2result(det_bboxes, det_coefs, det_labels, self.bbox_head.num_classes, img_meta[0],
                                 self.bases, self.method, x_mean_32, sqrt_var_32)
                for det_bboxes, det_labels, det_coefs in bbox_list]

        elif self.method == 'cosine':
            results = [
                bbox_mask2result(det_bboxes, det_coefs, det_labels, self.bbox_head.num_classes, img_meta[0],
                                 self.bases, self.method)
                for det_bboxes, det_labels, det_coefs in bbox_list]

        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results
