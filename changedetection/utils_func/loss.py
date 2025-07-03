import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, einsum
import torch.nn .functional as F
from typing import Iterable, Set, Tuple
from scipy.ndimage import distance_transform_edt
import torchvision.models as models

from RemoteSensing.changedetection.utils_func.utils import simplex, class2one_hot, uniq

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def boundary_loss(pred, target):
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    boundary_distances = []
    for b in range(target.shape[0]):  # Iterate over batch
        for c in range(target_onehot.shape[1]):  # Iterate over classes
            mask = target_onehot[b, c].cpu().numpy()  # Get binary mask for class c
            if np.sum(mask) == 0:  # Skip if no pixels for this class
                boundary_distances.append(np.zeros_like(mask))
                continue
            # Compute the distance transform for the foreground (class c)
            pos_dist = distance_transform_edt(mask)
            neg_dist = distance_transform_edt(1 - mask)
            boundary_dist = pos_dist + neg_dist
            boundary_distances.append(boundary_dist)
    
    boundary_distances = np.stack(boundary_distances, axis=0)  # [B*C, H, W]
    boundary_distances = torch.from_numpy(boundary_distances).float().to(pred.device)  # Move to GPU if needed

    boundary_distances = boundary_distances.view(pred.shape)  # [B, C, H, W]

    # Compute the boundary loss
    pred_softmax = F.softmax(pred, dim=1)  # Convert logits to probabilities
    loss = torch.mean(pred_softmax * boundary_distances)  # Weighted sum

    return loss


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.reshape(-1)
    truth = truth_pixel.reshape(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss

def dice_loss(predicts,target,weight=None):
    idc= [0, 1]
    probs = torch.softmax(predicts, dim=1)
    # target = target.unsqueeze(1)
    target = class2one_hot(target, 7)
    assert simplex(probs) and simplex(target)

    pc = probs[:, idc, ...].type(torch.float32)
    tc = target[:, idc, ...].type(torch.float32)
    intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
    union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

    divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

    loss = divided.mean()
    return loss


def dice_loss_multiclass(pred, target, smooth=1e-6, ignore_index=255):
    # Mask out invalid pixels
    valid_mask = (target != ignore_index).float()
    target = target.clone()
    target[target == ignore_index] = 0
    
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Apply valid_mask
    pred = pred * valid_mask.unsqueeze(1)
    target_one_hot = target_one_hot * valid_mask.unsqueeze(1)
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def ce_dice(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def dice(input, target, weight=None):
    dice_loss_ = dice_loss(input, target)
    return dice_loss_

def ce2_dice1(input, target, ignore_index=255):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    labels_bn = (target > 0).float()  # Binary labels (0 or 1)

    logits_positive = input[:, 1, :, :]  # Shape: [N, H, W]

    bce_loss = weighted_BCE_logits(logits_positive, labels_bn)
    loss = 1 * ce_loss + 0.15* dice_loss_ + 0.35 * bce_loss
    return loss

def ce2_dice1_multiclass(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    target2 = target.clone()
    dice_loss_ = dice_loss_multiclass(input, target2)
    loss = ce_loss + 0.25 * dice_loss_ 
    return loss


def ce1_dice2(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss +  dice_loss_
    return loss

def ce_scl(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def contrastive_loss(features_1, features_2, label, margin=1.0):
    similarity = F.cosine_similarity(features_1, features_2, dim=1)

    # Loss for unchanged regions (maximize similarity)
    unchanged_loss = (1 - similarity) * (1 - label)  # Mask for unchanged regions

    # Loss for changed regions (minimize similarity)
    changed_loss = torch.clamp(similarity - margin, min=0) * label  # Mask for changed regions

    # Combine losses
    loss = unchanged_loss.mean() + changed_loss.mean()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=255)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16].cuda().eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return F.mse_loss(input_features, target_features)


def tversky_loss(output, target, alpha=0.7, beta=0.3, smooth=1e-6):
    # Binary change detection assumed (classes 0: no change, 1: change)
    logits = F.softmax(output, dim=1)
    pred = logits[:, 1, ...]  # Probability of change class
    target = (target == 1).float()  # Convert to binary mask
    
    # Flatten tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate TP, FP, FN
    tp = (pred_flat * target_flat).sum()
    fp = ((1 - target_flat) * pred_flat).sum()
    fn = (target_flat * (1 - pred_flat)).sum()
    
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - tversky


class SeK_Loss(nn.Module):
    def __init__(self, num_classes, non_change_class=0, beta=1.5, gamma=0.5, eps=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.non_change = non_change_class
        self.beta = beta  # Controls IoU emphasis
        self.gamma = gamma  # Class balancing
        self.eps = eps
        
    def forward(self, pred_t1, pred_t2, label_t1, label_t2, change_mask):
        """
        Args:
            pred_t1: (B, C, H, W) logits for time 1
            pred_t2: (B, C, H, W) logits for time 2
            label_t1: (B, H, W) ground truth labels T1
            label_t2: (B, H, W) ground truth labels T2
            change_mask: (B, H, W) binary mask (1=changed)
        """
        B, C, H, W = pred_t1.shape
        device = pred_t1.device
        
        # Mask changed regions
        change_mask = change_mask.unsqueeze(1)  # B,1,H,W
        valid_classes = [c for c in range(C) if c != self.non_change]
        
        # Convert to probabilities
        prob_t1 = F.softmax(pred_t1, dim=1) * change_mask
        prob_t2 = F.softmax(pred_t2, dim=1) * change_mask
        
        # Flatten tensors
        prob_t1 = prob_t1.permute(0,2,3,1).reshape(-1, C)  # (N, C)
        prob_t2 = prob_t2.permute(0,2,3,1).reshape(-1, C)
        label_t1 = label_t1.reshape(-1)  # (N)
        label_t2 = label_t2.reshape(-1)
        mask = change_mask.reshape(-1).bool()  # (N)
        
        # Filter changed pixels
        prob_t1 = prob_t1[mask]
        prob_t2 = prob_t2[mask]
        label_t1 = label_t1[mask]
        label_t2 = label_t2[mask]
        
        if prob_t1.size(0) == 0:  # No changes in batch
            return torch.tensor(0.0).to(device)
            
        # 1. Kappa Component --------------------------------------------------
        def compute_kappa(probs, labels):
            # Build soft confusion matrix
            oh_labels = F.one_hot(labels, C).float()  # (N, C)
            conf_matrix = torch.matmul(probs.T, oh_labels)  # (C, C)
            
            # Exclude non-change class
            conf_matrix = conf_matrix[valid_classes][:, valid_classes]
            total = conf_matrix.sum()
            
            # Observed agreement
            po = torch.diag(conf_matrix).sum() / total
            
            # Expected agreement
            row_sum = conf_matrix.sum(dim=1)
            col_sum = conf_matrix.sum(dim=0)
            pe = torch.sum(row_sum * col_sum) / (total ** 2)
            
            return (po - pe) / (1 - pe + self.eps)
            
        kappa_t1 = compute_kappa(prob_t1, label_t1)
        kappa_t2 = compute_kappa(prob_t2, label_t2)
        kappa = (kappa_t1 + kappa_t2) / 2
        
        # 2. mIoU Component ---------------------------------------------------
        def compute_iou(probs, labels):
            oh_labels = F.one_hot(labels, C).float()
            intersection = (probs * oh_labels).sum(dim=0)[valid_classes]
            union = probs.sum(dim=0)[valid_classes] + oh_labels.sum(dim=0)[valid_classes] - intersection
            
            # Frequency weighting
            freq = oh_labels.sum(dim=0)[valid_classes]
            weights = 1 / torch.log(freq + 1 + self.eps)
            weights = weights / weights.sum()
            
            return (intersection / (union + self.eps) * weights).sum()
            
        iou_t1 = compute_iou(prob_t1, label_t1)
        iou_t2 = compute_iou(prob_t2, label_t2)
        miou = (iou_t1 + iou_t2) / 2
        
        # 3. Combined Loss ----------------------------------------------------
        sek_value = kappa * torch.exp(self.beta * miou)
        loss = 1 - sek_value + self.gamma * (1 - miou)
        
        return torch.clip(loss, min=0.0)