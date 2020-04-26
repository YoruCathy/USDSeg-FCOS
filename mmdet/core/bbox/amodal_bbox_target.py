import torch

from ..utils import multi_apply
from .transforms import bbox2delta


def amodal_bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_coefs_list,
                neg_coefs_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                pos_gt_coefs_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    # print(pos_coefs_list)
    labels, label_weights, bbox_targets, bbox_weights, coef_targets, coef_weights = multi_apply(
        amodal_bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_coefs_list,
        neg_coefs_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        pos_gt_coefs_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        coef_targets = torch.cat(coef_targets,0)
        coef_weights = torch.cat(coef_weights,0)
    return labels, label_weights, bbox_targets, bbox_weights, coef_targets, coef_weights


def amodal_bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_coefs,
                       neg_coefs,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       pos_gt_coefs,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    coef_targets = pos_bboxes.new_zeros(num_samples, 32)
    coef_weights = pos_bboxes.new_zeros(num_samples, 32)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)

        bbox_targets[:num_pos, :] = pos_bbox_targets
        coef_targets[:num_pos, :] = pos_gt_coefs
        bbox_weights[:num_pos, :] = 1
        coef_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights, coef_targets, coef_weights


def amodal_expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand