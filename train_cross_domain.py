"""
多模态跨域自适应UNet训练脚本
正确的代码组织结构
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from mmwhs_dataset import create_mmwhs_loaders
warnings.filterwarnings('ignore')


# ==================== 1. 梯度反转层（最基础的工具） ====================
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


# ==================== 2. UNet基础组件 ====================
class DoubleConv(nn.Module):
    """双卷积层"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块（最终修复版）"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 🔥 拼接后通道数 = in_channels(下层上采样) + out_channels(跳跃连接)
            # 但上采样前in_channels通道会减半
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: 来自下层 (需要上采样)
        x2: 跳跃连接
        """
        x1 = self.up(x1)

        # 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ==================== 3. 核心模块（完整版） ====================

class SemanticAdaptiveFusion(nn.Module):
    """语义自适应融合模块"""

    def __init__(self, channels):
        super().__init__()

        # 语义特征提取
        self.ct_semantic = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.mri_semantic = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 相似度计算
        self.similarity = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, ct_feat, mri_feat):
        """
        自适应融合

        Returns:
            fused: 融合后的特征
        """
        # 语义特征
        ct_sem = self.ct_semantic(ct_feat)
        mri_sem = self.mri_semantic(mri_feat)

        # 计算相似度权重
        concat = torch.cat([ct_sem, mri_sem], dim=1)
        weights = self.similarity(concat)  # [B, 2, H, W]

        # 加权融合
        ct_weight = weights[:, 0:1, :, :]
        mri_weight = weights[:, 1:2, :, :]

        fused = ct_feat * ct_weight + mri_feat * mri_weight

        return fused


class CrossDomainAlignment(nn.Module):
    """跨域对齐模块"""

    def __init__(self, channels):
        super().__init__()

        # 域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels // 2, 1)
        )

        # 特征对齐
        self.align_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ct_feat, mri_feat):
        """
        跨域对齐

        Returns:
            ct_aligned: 对齐后的CT特征
            mri_aligned: 对齐后的MRI特征
            domain_loss: 域对抗损失
        """
        # 特征对齐
        ct_aligned = self.align_conv(ct_feat)
        mri_aligned = self.align_conv(mri_feat)

        if self.training:
            # 梯度反转
            ct_reversed = GradientReversalLayer.apply(ct_aligned, 1.0)
            mri_reversed = GradientReversalLayer.apply(mri_aligned, 1.0)

            # 域判别
            ct_domain_pred = self.domain_discriminator(ct_reversed)
            mri_domain_pred = self.domain_discriminator(mri_reversed)

            # 域对抗损失
            ct_labels = torch.zeros_like(ct_domain_pred)
            mri_labels = torch.ones_like(mri_domain_pred)

            bce_loss = nn.BCEWithLogitsLoss()
            domain_loss = (bce_loss(ct_domain_pred, ct_labels) +
                           bce_loss(mri_domain_pred, mri_labels)) / 2
        else:
            domain_loss = torch.tensor(0.0, device=ct_feat.device)

        return ct_aligned, mri_aligned, domain_loss


class MultiModalPriorMask(nn.Module):
    """多模态先验掩码生成器"""

    def __init__(self, channels):
        super().__init__()

        self.prior_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, ct_feat, mri_feat):
        """
        生成先验掩码

        Returns:
            prior: 先验掩码 [B, 1, H, W]
        """
        concat = torch.cat([ct_feat, mri_feat], dim=1)
        prior = self.prior_net(concat)
        return prior


class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==================== 4. 完整模型 ====================
class MultiModalCrossDomainUNet(nn.Module):
    """多模态跨域自适应UNet - 完整版"""

    def __init__(self, n_classes=2, bilinear=True, base_channels=64):
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        # CT编码器
        self.ct_inc = DoubleConv(1, base_channels)
        self.ct_down1 = Down(base_channels, base_channels * 2)
        self.ct_down2 = Down(base_channels * 2, base_channels * 4)
        self.ct_down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.ct_down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # MRI编码器
        self.mri_inc = DoubleConv(1, base_channels)
        self.mri_down1 = Down(base_channels, base_channels * 2)
        self.mri_down2 = Down(base_channels * 2, base_channels * 4)
        self.mri_down3 = Down(base_channels * 4, base_channels * 8)
        self.mri_down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # 融合模块
        self.fusion_modules = nn.ModuleList([
            SemanticAdaptiveFusion(base_channels),
            SemanticAdaptiveFusion(base_channels * 2),
            SemanticAdaptiveFusion(base_channels * 4),
            SemanticAdaptiveFusion(base_channels * 8),
            SemanticAdaptiveFusion(base_channels * 16 // factor)
        ])

        # 域对齐模块
        self.alignment_modules = nn.ModuleList([
            CrossDomainAlignment(base_channels),
            CrossDomainAlignment(base_channels * 2),
            CrossDomainAlignment(base_channels * 4),
            CrossDomainAlignment(base_channels * 8),
            CrossDomainAlignment(base_channels * 16 // factor)
        ])

        # 先验生成器
        self.prior_generator = MultiModalPriorMask(base_channels * 16 // factor)

        # 🔥 解码器 - 修正通道数计算
        # Up(输入通道, 输出通道)
        # 输入通道 = 下层特征 + 跳跃连接特征
        self.up1 = Up(base_channels * 16 // factor + base_channels * 8, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8 // factor + base_channels * 4, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4 // factor + base_channels * 2, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2 // factor + base_channels, base_channels, bilinear)

        # 输出层
        self.ct_outc = OutConv(base_channels, n_classes)
        self.mri_outc = OutConv(base_channels, n_classes)

        # 保存输出
        self.ct_output = None
        self.mri_output = None

    def forward(self, ct_img, mri_img, return_details=False):
        """前向传播 - 完全修复版"""

        # CT编码
        ct1 = self.ct_inc(ct_img)  # [B, 32, 256, 256]
        ct2 = self.ct_down1(ct1)  # [B, 64, 128, 128]
        ct3 = self.ct_down2(ct2)  # [B, 128, 64, 64]
        ct4 = self.ct_down3(ct3)  # [B, 256, 32, 32]
        ct5 = self.ct_down4(ct4)  # [B, 256, 16, 16]

        # MRI编码
        mri1 = self.mri_inc(mri_img)
        mri2 = self.mri_down1(mri1)
        mri3 = self.mri_down2(mri2)
        mri4 = self.mri_down3(mri3)
        mri5 = self.mri_down4(mri4)

        # 融合和对齐
        ct_feats = [ct1, ct2, ct3, ct4, ct5]
        mri_feats = [mri1, mri2, mri3, mri4, mri5]

        total_align_loss = 0.0
        fused_feats = []

        for i, (fusion, align) in enumerate(zip(self.fusion_modules, self.alignment_modules)):
            fused = fusion(ct_feats[i], mri_feats[i])
            ct_aligned, mri_aligned, domain_loss = align(ct_feats[i], mri_feats[i])
            total_align_loss += domain_loss
            fused_feats.append(fused)

        align_loss = total_align_loss / len(self.fusion_modules)

        # 🔥 先验掩码生成（底层特征）
        ct_prior_low = self.prior_generator(ct5, mri5)  # [B, 1, 16, 16]
        mri_prior_low = self.prior_generator(mri5, ct5)

        # CT解码
        ct_x = self.up1(fused_feats[4], fused_feats[3])  # 256->128
        ct_x = self.up2(ct_x, fused_feats[2])  # 128->64
        ct_x = self.up3(ct_x, fused_feats[1])  # 64->32
        ct_x = self.up4(ct_x, fused_feats[0])  # 32->32

        # MRI解码
        mri_x = self.up1(fused_feats[4], fused_feats[3])
        mri_x = self.up2(mri_x, fused_feats[2])
        mri_x = self.up3(mri_x, fused_feats[1])
        mri_x = self.up4(mri_x, fused_feats[0])

        # 输出
        self.ct_output = self.ct_outc(ct_x)
        self.mri_output = self.mri_outc(mri_x)

        # 🔥 删除 prior 生成和 prior_loss 计算
        # 只返回对齐损失
        seg_loss_ct = torch.tensor(0.0, device=ct_img.device)
        seg_loss_mri = torch.tensor(0.0, device=mri_img.device)
        prior_loss = torch.tensor(0.0, device=ct_img.device)

        # 🔥 只返回 prior_loss 和 align_loss，seg_loss在外部计算
        return seg_loss_ct, seg_loss_mri, prior_loss, align_loss

# ==================== 5. 损失函数 ====================
class DiceLoss(nn.Module):
    """Dice损失"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 1e-5
        input_soft = F.softmax(input, dim=1)
        target_one_hot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for i in range(self.n_classes):
            input_i = input_soft[:, i, :, :]
            target_i = target_one_hot[:, i, :, :]

            intersection = (input_i * target_i).sum()
            union = input_i.sum() + target_i.sum()

            dice = (2. * intersection + smooth) / (union + smooth)
            dice_per_class.append(dice)

        return 1 - torch.mean(torch.stack(dice_per_class))


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] - 预测logits
            target: [B, H, W] - 真实标签（整数）
        """
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)

        # One-hot编码
        target_one_hot = torch.zeros_like(log_preds).scatter(1, target.unsqueeze(1), 1)

        # 标签平滑
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # 计算损失
        loss = (-target_smooth * log_preds).sum(dim=1).mean()
        return loss

class MultiModalCrossDomainLoss(nn.Module):
    def __init__(self, n_classes, seg_weight=1.0, prior_weight=0.3, align_weight=0.01):
        super().__init__()
        self.seg_weight = seg_weight
        self.prior_weight = prior_weight
        self.align_weight = 0.01  # ← 改为0.01（减少10倍）

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes)

    def forward(self, outputs, target):
        """
        计算多任务损失
        """
        # 1. 主分割损失（已经是原始尺寸）
        seg_ce = self.ce_loss(outputs['segmentation'], target)
        seg_dice = self.dice_loss(outputs['segmentation'], target)
        seg_loss = 0.4 * seg_ce + 0.6 * seg_dice

        # 🔥 修复：将先验掩码上采样到原始尺寸
        fused_prior = outputs['prior_masks']['fused_prior']
        ct_prior = outputs['prior_masks']['ct_prior']
        mri_prior = outputs['prior_masks']['mri_prior']

        # 上采样到和target相同的尺寸
        target_size = target.shape[-2:]  # (H, W)

        fused_prior_upsampled = F.interpolate(
            fused_prior,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )

        ct_prior_upsampled = F.interpolate(
            ct_prior,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )

        mri_prior_upsampled = F.interpolate(
            mri_prior,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )

        # 2. 先验掩码监督损失（使用上采样后的掩码）
        prior_ce = self.ce_loss(fused_prior_upsampled, target)
        prior_dice = self.dice_loss(fused_prior_upsampled, target)
        prior_loss = 0.4 * prior_ce + 0.6 * prior_dice

        # 3. CT和MRI各自的先验损失
        ct_prior_loss = self.dice_loss(ct_prior_upsampled, target)
        mri_prior_loss = self.dice_loss(mri_prior_upsampled, target)

        # 4. 域对齐损失
        alignment_losses = outputs['alignment_losses']
        alignment_loss = sum(alignment_losses) / len(alignment_losses)

        # 5. 总损失
        total_loss = (
                self.seg_weight * seg_loss +
                self.prior_weight * (prior_loss + 0.2 * ct_prior_loss + 0.2 * mri_prior_loss) +
                self.align_weight * alignment_loss
        )

        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'prior_loss': prior_loss,
            'alignment_loss': alignment_loss,
            'seg_ce': seg_ce,
            'seg_dice': seg_dice
        }


class BoundaryLoss(nn.Module):
    """边界损失（提高IoU）"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2, H, W] - 预测logits
            target: [B, H, W] - 真实标签
        """
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # 预测边界
        pred_prob = torch.softmax(pred, dim=1)[:, 1:2, :, :]  # [B, 1, H, W]
        pred_edge_x = F.conv2d(pred_prob, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_prob, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-8)

        # 真实边界
        target_float = target.unsqueeze(1).float()
        target_edge_x = F.conv2d(target_float, sobel_x, padding=1)
        target_edge_y = F.conv2d(target_float, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-8)

        # MSE损失
        return F.mse_loss(pred_edge, target_edge)

# ==================== 6. 数据集类 ====================
class MultiModalDataset(Dataset):
    """多模态数据集"""
    def __init__(self, data_root, split='train', augment=False):
        self.data_root = Path(data_root)
        self.split = split
        self.augment = augment

        self.ct_img_dir = self.data_root / split / 'ct' / 'images'
        self.ct_mask_dir = self.data_root / split / 'ct' / 'masks'
        self.mri_img_dir = self.data_root / split / 'mri' / 'images'
        self.mri_mask_dir = self.data_root / split / 'mri' / 'masks'

        self.ct_files = sorted(self.ct_img_dir.glob('*.png'))

        if len(self.ct_files) == 0:
            raise ValueError(f"未找到CT图像: {self.ct_img_dir}")

        print(f"    {split:8s}: {len(self.ct_files)} 个样本")

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct_img_path = self.ct_files[idx]
        ct_img = np.array(Image.open(ct_img_path), dtype=np.float32) / 255.0
        ct_mask = np.array(Image.open(self.ct_mask_dir / ct_img_path.name), dtype=np.int64)

        mri_img_path = self.mri_img_dir / ct_img_path.name
        mri_mask_path = self.mri_mask_dir / ct_img_path.name

        if mri_img_path.exists():
            mri_img = np.array(Image.open(mri_img_path), dtype=np.float32) / 255.0
            mri_mask = np.array(Image.open(mri_mask_path), dtype=np.int64)
        else:
            mri_img = np.zeros_like(ct_img, dtype=np.float32)
            mri_mask = np.zeros_like(ct_mask, dtype=np.int64)

        if self.augment:
            ct_img, ct_mask, mri_img, mri_mask = self.apply_augmentation(
                ct_img, ct_mask, mri_img, mri_mask
            )

        ct_img = torch.from_numpy(np.ascontiguousarray(ct_img[None, :, :])).float()
        mri_img = torch.from_numpy(np.ascontiguousarray(mri_img[None, :, :])).float()
        ct_mask = torch.from_numpy(np.ascontiguousarray(ct_mask)).long()
        mri_mask = torch.from_numpy(np.ascontiguousarray(mri_mask)).long()

        return {
            'ct_image': ct_img,
            'mri_image': mri_img,
            'ct_mask': ct_mask,
            'mri_mask': mri_mask
        }

    def apply_augmentation(self, ct_img, ct_mask, mri_img, mri_mask):
        """
        数据增强（修复版）

        Args:
            ct_img: CT图像 [H, W]
            ct_mask: CT掩码 [H, W]
            mri_img: MRI图像 [H, W]
            mri_mask: MRI掩码 [H, W]
        """
        # 1. 水平翻转
        if random.random() > 0.5:
            ct_img = np.fliplr(ct_img)
            ct_mask = np.fliplr(ct_mask)
            mri_img = np.fliplr(mri_img)
            mri_mask = np.fliplr(mri_mask)

        # 2. 旋转
        if random.random() > 0.5:
            k = random.randint(1, 3)
            ct_img = np.rot90(ct_img, k)
            ct_mask = np.rot90(ct_mask, k)
            mri_img = np.rot90(mri_img, k)
            mri_mask = np.rot90(mri_mask, k)

        # 3. 亮度调整
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            ct_img = np.clip(ct_img * factor, 0, 1)
            mri_img = np.clip(mri_img * factor, 0, 1)

        # 4. 高斯噪声
        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.03)
            ct_noise = np.random.randn(*ct_img.shape) * noise_std
            mri_noise = np.random.randn(*mri_img.shape) * noise_std
            ct_img = np.clip(ct_img + ct_noise, 0, 1)
            mri_img = np.clip(mri_img + mri_noise, 0, 1)

        # 5. 对比度调整（可选）
        if random.random() > 0.7:
            # Gamma变换
            gamma = random.uniform(0.8, 1.2)
            ct_img = np.power(ct_img, gamma)
            mri_img = np.power(mri_img, gamma)

        return ct_img, ct_mask, mri_img, mri_mask


# ==================== 7. 评估指标 ====================
def calculate_metrics(pred, target, n_classes):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}

    for class_id in range(1, n_classes):
        pred_mask = (pred == class_id).astype(np.float32)
        true_mask = (target == class_id).astype(np.float32)

        tp = (pred_mask * true_mask).sum()
        fp = (pred_mask * (1 - true_mask)).sum()
        fn = ((1 - pred_mask) * true_mask).sum()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        metrics['dice'].append(dice)

        iou = tp / (tp + fp + fn + 1e-8)
        metrics['iou'].append(iou)

        precision = tp / (tp + fp + 1e-8)
        metrics['precision'].append(precision)

        recall = tp / (tp + fn + 1e-8)
        metrics['recall'].append(recall)

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


# ==================== TTA工具函数 ====================

@torch.no_grad()
def predict_with_tta(model, ct_img, mri_img, device, tta_mode='full'):
    """
    测试时增强预测

    Args:
        model: 训练好的模型
        ct_img: CT图像 [B, 1, H, W]
        mri_img: MRI图像 [B, 1, H, W]
        device: 设备
        tta_mode: 增强模式
            - 'none': 不使用TTA
            - 'basic': 基础TTA（翻转，2倍速度）
            - 'full': 完整TTA（翻转+旋转，4倍速度）

    Returns:
        pred: 平均后的预测 [B, n_classes, H, W]
    """
    model.eval()
    predictions = []

    if tta_mode == 'none':
        # 不使用TTA，直接预测
        pred = model(ct_img, mri_img, return_details=False)
        return pred

    # 1. 原始图像
    pred = model(ct_img, mri_img, return_details=False)
    predictions.append(pred)

    # 2. 水平翻转
    ct_flip_h = torch.flip(ct_img, dims=[-1])
    mri_flip_h = torch.flip(mri_img, dims=[-1])
    pred_flip_h = model(ct_flip_h, mri_flip_h, return_details=False)
    pred_flip_h = torch.flip(pred_flip_h, dims=[-1])  # 翻转回来
    predictions.append(pred_flip_h)

    if tta_mode == 'basic':
        # 基础模式：只用翻转
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred

    # 3. 垂直翻转
    ct_flip_v = torch.flip(ct_img, dims=[-2])
    mri_flip_v = torch.flip(mri_img, dims=[-2])
    pred_flip_v = model(ct_flip_v, mri_flip_v, return_details=False)
    pred_flip_v = torch.flip(pred_flip_v, dims=[-2])
    predictions.append(pred_flip_v)

    # 4. 旋转90度
    ct_rot90 = torch.rot90(ct_img, k=1, dims=[-2, -1])
    mri_rot90 = torch.rot90(mri_img, k=1, dims=[-2, -1])
    pred_rot90 = model(ct_rot90, mri_rot90, return_details=False)
    pred_rot90 = torch.rot90(pred_rot90, k=-1, dims=[-2, -1])  # 旋转回来
    predictions.append(pred_rot90)

    # 5. 旋转180度
    ct_rot180 = torch.rot90(ct_img, k=2, dims=[-2, -1])
    mri_rot180 = torch.rot90(mri_img, k=2, dims=[-2, -1])
    pred_rot180 = model(ct_rot180, mri_rot180, return_details=False)
    pred_rot180 = torch.rot90(pred_rot180, k=-2, dims=[-2, -1])
    predictions.append(pred_rot180)

    # 6. 旋转270度
    ct_rot270 = torch.rot90(ct_img, k=3, dims=[-2, -1])
    mri_rot270 = torch.rot90(mri_img, k=3, dims=[-2, -1])
    pred_rot270 = model(ct_rot270, mri_rot270, return_details=False)
    pred_rot270 = torch.rot90(pred_rot270, k=-3, dims=[-2, -1])
    predictions.append(pred_rot270)

    # 平均所有预测
    final_pred = torch.stack(predictions).mean(dim=0)

    return final_pred

# ==================== 8. 训练器 ====================
class CrossDomainTrainer:
    """跨域训练器"""

    def __init__(self, model, train_loader, val_loader, config, save_dir):  # ✅ 添加 save_dir 参数
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config['device']
        self.save_dir = Path(save_dir)  # ✅ 现在可以正常使用了
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )

        # 🔥 学习率调度器 - 更长warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config['warmup_epochs']  # 8轮
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'] - config['warmup_epochs'],  # 120-8=112
            eta_min=config['min_lr']  # 5e-7
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config['warmup_epochs']]
        )

        # 🔥 调整后的损失权重
        # 🔥 平衡的损失权重（方案B的折中）
        self.criterion = MultiModalCrossDomainLoss(
            n_classes=config['n_classes'],
            seg_weight=0.55,  # 🔥 折中值（0.5和0.6之间）
            prior_weight=0.25,  # 🔥 折中值（0.2和0.3之间）
            align_weight=0.20  # 保持
        )
        # 🔥 添加标签平滑损失（新增）
        self.smooth_ce = LabelSmoothingCrossEntropy(smoothing=0.05).to(self.device)

        # 🔥 添加边界损失（新增）
        self.boundary_loss = BoundaryLoss().to(self.device)

        self.use_amp = config.get('use_amp', True)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("  ✅ 启用混合精度训练")
        else:
            self.scaler = None

        self.history = {
            'train_loss': [],
            'train_seg_loss': [],
            'train_prior_loss': [],
            'train_align_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],  # ← 新增
            'val_recall': [],  # ← 新增
            'learning_rate': [],
            'epoch_time': []
        }

        self.best_dice = 0.0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_start = time.time()

        total_loss = 0
        total_seg = 0
        total_prior = 0
        total_align = 0
        total_boundary = 0  # 🔥 新增

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}',
                    bar_format='{l_bar}{bar:30}{r_bar}')
        self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            ct_img = batch['ct_image'].to(self.device)
            mri_img = batch['mri_image'].to(self.device)
            ct_mask = batch['ct_mask'].to(self.device)
            mri_mask = batch['mri_mask'].to(self.device)
            ct_prior_gt = batch['ct_prior'].to(self.device)
            mri_prior_gt = batch['mri_prior'].to(self.device)

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    # 前向传播
                    _, _, _, align_loss = self.model(ct_img, mri_img)

                    ct_pred = self.model.ct_output
                    mri_pred = self.model.mri_output

                    # 🔥 使用标签平滑（替换 F.cross_entropy）
                    seg_loss_ct = self.smooth_ce(ct_pred, ct_mask)
                    seg_loss_mri = self.smooth_ce(mri_pred, mri_mask)
                    seg_loss = (seg_loss_ct + seg_loss_mri) / 2

                    # Prior损失
                    ct_pred_prob = torch.softmax(ct_pred, dim=1)[:, 1:2, :, :]
                    mri_pred_prob = torch.softmax(mri_pred, dim=1)[:, 1:2, :, :]
                    prior_loss = (F.mse_loss(ct_pred_prob, ct_prior_gt) +
                                  F.mse_loss(mri_pred_prob, mri_prior_gt)) / 2

                    # 🔥 边界损失（使用 self.boundary_loss）
                    boundary_loss = (self.boundary_loss(ct_pred, ct_mask) +
                                     self.boundary_loss(mri_pred, mri_mask)) / 2

                    # 🔥 总损失（加入边界损失）
                    loss = (self.criterion.seg_weight * seg_loss +
                            self.criterion.prior_weight * prior_loss +
                            self.criterion.align_weight * align_loss +
                            0.1 * boundary_loss)  # 边界损失权重0.1

                    loss = loss / self.config['accumulation_steps']

                # 反向传播 + 梯度裁剪
                self.scaler.scale(loss).backward()

                if (i + 1) % self.config['accumulation_steps'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            else:
                # 非混合精度训练
                _, _, _, align_loss = self.model(ct_img, mri_img)

                ct_pred = self.model.ct_output
                mri_pred = self.model.mri_output

                # 🔥 使用标签平滑
                seg_loss_ct = self.smooth_ce(ct_pred, ct_mask)
                seg_loss_mri = self.smooth_ce(mri_pred, mri_mask)
                seg_loss = (seg_loss_ct + seg_loss_mri) / 2

                # Prior损失
                ct_pred_prob = torch.softmax(ct_pred, dim=1)[:, 1:2, :, :]
                mri_pred_prob = torch.softmax(mri_pred, dim=1)[:, 1:2, :, :]
                prior_loss = (F.mse_loss(ct_pred_prob, ct_prior_gt) +
                              F.mse_loss(mri_pred_prob, mri_prior_gt)) / 2

                # 🔥 边界损失
                boundary_loss = (self.boundary_loss(ct_pred, ct_mask) +
                                 self.boundary_loss(mri_pred, mri_mask)) / 2

                # 总损失
                loss = (self.criterion.seg_weight * seg_loss +
                        self.criterion.prior_weight * prior_loss +
                        self.criterion.align_weight * align_loss +
                        0.1 * boundary_loss)

                loss = loss / self.config['accumulation_steps']
                loss.backward()

                if (i + 1) % self.config['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # 累计损失
            total_loss += loss.item() * self.config['accumulation_steps']
            total_seg += seg_loss.item()
            total_prior += prior_loss.item()
            total_align += align_loss.item()
            total_boundary += boundary_loss.item()  # 🔥 新增

            pbar.set_postfix({
                'loss': f'{loss.item() * self.config["accumulation_steps"]:.4f}',
                'seg': f'{seg_loss.item():.4f}',
                'prior': f'{prior_loss.item():.4f}',
                'align': f'{align_loss.item():.4f}',
                'bound': f'{boundary_loss.item():.4f}'  # 🔥 显示边界损失
            })

        epoch_time = time.time() - epoch_start

        return {
                   'total': total_loss / len(self.train_loader),
                   'seg': total_seg / len(self.train_loader),
                   'prior': total_prior / len(self.train_loader),
                   'align': total_align / len(self.train_loader),
                   'boundary': total_boundary / len(self.train_loader)  # 🔥 新增
               }, epoch_time

    def _tta_forward(self, ct_img, mri_img, mode='simple'):
        """
        测试时增强(TTA)前向传播

        Returns:
            ct_pred: [B, 2, H, W]
            mri_pred: [B, 2, H, W]
        """
        ct_predictions = []
        mri_predictions = []

        # 1. 原图
        with torch.no_grad():
            self.model(ct_img, mri_img)
            ct_predictions.append(self.model.ct_output.clone())
            mri_predictions.append(self.model.mri_output.clone())

        if mode == 'full':
            # 2. 水平翻转
            ct_flip = torch.flip(ct_img, dims=[-1])
            mri_flip = torch.flip(mri_img, dims=[-1])
            self.model(ct_flip, mri_flip)
            ct_predictions.append(torch.flip(self.model.ct_output, dims=[-1]))
            mri_predictions.append(torch.flip(self.model.mri_output, dims=[-1]))

            # 3. 垂直翻转
            ct_vflip = torch.flip(ct_img, dims=[-2])
            mri_vflip = torch.flip(mri_img, dims=[-2])
            self.model(ct_vflip, mri_vflip)
            ct_predictions.append(torch.flip(self.model.ct_output, dims=[-2]))
            mri_predictions.append(torch.flip(self.model.mri_output, dims=[-2]))

            # 4. 旋转90度
            ct_rot90 = torch.rot90(ct_img, k=1, dims=[-2, -1])
            mri_rot90 = torch.rot90(mri_img, k=1, dims=[-2, -1])
            self.model(ct_rot90, mri_rot90)
            ct_predictions.append(torch.rot90(self.model.ct_output, k=-1, dims=[-2, -1]))
            mri_predictions.append(torch.rot90(self.model.mri_output, k=-1, dims=[-2, -1]))

        # 平均所有预测
        ct_final = torch.stack(ct_predictions).mean(dim=0)  # [B, 2, H, W]
        mri_final = torch.stack(mri_predictions).mean(dim=0)

        return ct_final, mri_final

    @torch.no_grad()
    def validate(self, use_tta=False, tta_mode='simple'):
        """验证函数 - 完全修复版"""
        self.model.eval()

        total_loss = 0.0

        # CT和MRI分别统计
        ct_intersection = 0
        ct_union = 0
        ct_pred_sum = 0
        ct_gt_sum = 0
        ct_tp = 0
        ct_fp = 0
        ct_fn = 0

        mri_intersection = 0
        mri_union = 0
        mri_pred_sum = 0
        mri_gt_sum = 0
        mri_tp = 0
        mri_fp = 0
        mri_fn = 0

        pbar = tqdm(self.val_loader, desc='🔍 Validating', leave=False,
                    bar_format='{l_bar}{bar:30}{r_bar}')

        for batch in pbar:
            ct_img = batch['ct_image'].to(self.device)
            mri_img = batch['mri_image'].to(self.device)
            ct_mask = batch['ct_mask'].to(self.device)
            mri_mask = batch['mri_mask'].to(self.device)
            ct_prior_gt = batch['ct_prior'].to(self.device)
            mri_prior_gt = batch['mri_prior'].to(self.device)

            if use_tta:
                # TTA模式
                ct_pred_logits, mri_pred_logits = self._tta_forward(ct_img, mri_img, mode=tta_mode)
                loss = torch.tensor(0.0).to(self.device)
            else:
                # 正常验证
                _, _, _, align_loss = self.model(ct_img, mri_img)

                # 获取预测
                ct_pred_logits = self.model.ct_output
                mri_pred_logits = self.model.mri_output

                # 计算损失
                seg_loss_ct = F.cross_entropy(ct_pred_logits, ct_mask)
                seg_loss_mri = F.cross_entropy(mri_pred_logits, mri_mask)
                seg_loss = (seg_loss_ct + seg_loss_mri) / 2

                # Prior损失
                ct_pred_prob = torch.softmax(ct_pred_logits, dim=1)[:, 1:2, :, :]
                mri_pred_prob = torch.softmax(mri_pred_logits, dim=1)[:, 1:2, :, :]
                prior_loss = (F.mse_loss(ct_pred_prob, ct_prior_gt) +
                              F.mse_loss(mri_pred_prob, mri_prior_gt)) / 2

                loss = (self.criterion.seg_weight * seg_loss +
                        self.criterion.prior_weight * prior_loss +
                        self.criterion.align_weight * align_loss)

                total_loss += loss.item()

            # 🔥 获取预测类别（确保是Tensor）
            pred_ct = ct_pred_logits.argmax(dim=1)  # [B, H, W] - Tensor
            pred_mri = mri_pred_logits.argmax(dim=1)

            # 🔥 转为二值Tensor（不要转NumPy）
            pred_ct_binary = (pred_ct == 1).float()  # Tensor
            mask_ct_binary = (ct_mask == 1).float()

            # 展平
            pred_ct_binary = pred_ct_binary.reshape(-1)
            mask_ct_binary = mask_ct_binary.reshape(-1)

            # 累加指标
            ct_intersection += (pred_ct_binary * mask_ct_binary).sum().item()
            ct_union += ((pred_ct_binary + mask_ct_binary) > 0).float().sum().item()
            ct_pred_sum += pred_ct_binary.sum().item()
            ct_gt_sum += mask_ct_binary.sum().item()

            ct_tp += ((pred_ct_binary == 1) & (mask_ct_binary == 1)).float().sum().item()
            ct_fp += ((pred_ct_binary == 1) & (mask_ct_binary == 0)).float().sum().item()
            ct_fn += ((pred_ct_binary == 0) & (mask_ct_binary == 1)).float().sum().item()

            # 🔥 MRI指标
            pred_mri_binary = (pred_mri == 1).float()
            mask_mri_binary = (mri_mask == 1).float()

            pred_mri_binary = pred_mri_binary.reshape(-1)
            mask_mri_binary = mask_mri_binary.reshape(-1)

            mri_intersection += (pred_mri_binary * mask_mri_binary).sum().item()
            mri_union += ((pred_mri_binary + mask_mri_binary) > 0).float().sum().item()
            mri_pred_sum += pred_mri_binary.sum().item()
            mri_gt_sum += mask_mri_binary.sum().item()

            mri_tp += ((pred_mri_binary == 1) & (mask_mri_binary == 1)).float().sum().item()
            mri_fp += ((pred_mri_binary == 1) & (mask_mri_binary == 0)).float().sum().item()
            mri_fn += ((pred_mri_binary == 0) & (mask_mri_binary == 1)).float().sum().item()

        # 计算平均指标
        dice_ct = (2 * ct_intersection + 1e-8) / (ct_pred_sum + ct_gt_sum + 1e-8)
        dice_mri = (2 * mri_intersection + 1e-8) / (mri_pred_sum + mri_gt_sum + 1e-8)
        dice = (dice_ct + dice_mri) / 2

        iou_ct = (ct_intersection + 1e-8) / (ct_union + 1e-8)
        iou_mri = (mri_intersection + 1e-8) / (mri_union + 1e-8)
        iou = (iou_ct + iou_mri) / 2

        precision_ct = (ct_tp + 1e-8) / (ct_tp + ct_fp + 1e-8)
        precision_mri = (mri_tp + 1e-8) / (mri_tp + mri_fp + 1e-8)
        precision = (precision_ct + precision_mri) / 2

        recall_ct = (ct_tp + 1e-8) / (ct_tp + ct_fn + 1e-8)
        recall_mri = (mri_tp + 1e-8) / (mri_tp + mri_fn + 1e-8)
        recall = (recall_ct + recall_mri) / 2

        avg_loss = total_loss / len(self.val_loader) if not use_tta else 0.0

        return {
            'loss': avg_loss,
            'dice': dice,
            'dice_ct': dice_ct,
            'dice_mri': dice_mri,
            'iou': iou,
            'precision': precision,
            'recall': recall
        }




    def save_checkpoint(self, epoch, metrics, losses, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'losses': losses,
            'history': self.history,
            'config': self.config,
            'best_dice': self.best_dice
        }

        checkpoint_path = self.config['checkpoint_dir'] / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.config['checkpoint_dir'] / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"        ✅ 保存最佳模型! Dice={metrics['dice']:.4f}")

    def train(self):
        """训练主循环"""
        patience_counter = 0

        for epoch in range(1, self.config['epochs'] + 1):
            # 🔥 Warmup学习率调整（在训练epoch之前）
            if epoch <= self.config.get('warmup_epochs', 10):
                lr = self.config['lr'] * (epoch / self.config['warmup_epochs'])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            # 训练一个epoch
            train_losses, epoch_time = self.train_epoch(epoch)

            # 🔥 获取当前学习率（在调度器调整之前）
            current_lr = self.optimizer.param_groups[0]['lr']

            # 验证
            if epoch == 1 or epoch % 10 == 0:
                print(f"\n        🔍 使用TTA验证...")
                val_metrics = self.validate(use_tta=True, tta_mode='full')
            else:
                val_metrics = self.validate(use_tta=False)

            # 🔥 学习率调度（在获取current_lr之后）
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_seg_loss'].append(train_losses['seg'])
            self.history['train_prior_loss'].append(train_losses['prior'])
            self.history['train_align_loss'].append(train_losses['align'])

            # 🔥 边界损失（如果有的话）
            if 'boundary' in train_losses:
                if 'train_boundary_loss' not in self.history:
                    self.history['train_boundary_loss'] = []
                self.history['train_boundary_loss'].append(train_losses['boundary'])

            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics.get('precision', 0.0))
            self.history['val_recall'].append(val_metrics.get('recall', 0.0))
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            # 打印结果
            print(f"\n    Epoch {epoch}/{self.config['epochs']} ({epoch_time:.1f}s):")
            print(f"        Train Loss: {train_losses['total']:.4f} "
                  f"(Seg:{train_losses['seg']:.4f}, "
                  f"Prior:{train_losses['prior']:.4f}, "
                  f"Align:{train_losses['align']:.4f})")
            print(f"        Val Loss:   {val_metrics['loss']:.4f}")
            print(f"        Val Dice:   {val_metrics['dice']:.4f} {'🔥' if val_metrics['dice'] > 0.85 else ''}")
            print(f"        Val IoU:    {val_metrics['iou']:.4f}")
            print(f"        Precision:  {val_metrics.get('precision', 0.0):.4f}")
            print(f"        Recall:     {val_metrics.get('recall', 0.0):.4f}")
            print(f"        LR:         {current_lr:.6f}")  # ✅ 现在可以正常使用了

            # 判断是否是最佳模型
            current_dice = val_metrics['dice']
            is_best = current_dice > self.best_dice

            if is_best:
                improvement = current_dice - self.best_dice
                self.best_dice = current_dice
                patience_counter = 0
                print(f"        ✅ 新的最佳Dice: {current_dice:.4f} (↑{improvement:.4f})")
            else:
                patience_counter += 1
                print(
                    f"        ⏸️  未提升 (最佳: {self.best_dice:.4f}), patience: {patience_counter}/{self.config['patience']}")

            # 保存检查点
            self.save_checkpoint(epoch, val_metrics, train_losses, is_best)

            # 早停判断
            if patience_counter >= self.config['patience']:
                print(f"\n⏹️  早停触发! {self.config['patience']}轮未提升")
                break

        print(f"\n🎉 训练完成! 最佳Dice: {self.best_dice:.4f}")

        # 绘制训练曲线
        self.plot_history()

    def plot_history(self):
        try:
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))

            # 总损失
            axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
            axes[0, 0].plot(self.history['val_loss'], label='Val', linewidth=2)
            axes[0, 0].set_title('Total Loss', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

            # 分割损失
            axes[0, 1].plot(self.history['train_seg_loss'], label='Seg Loss', linewidth=2)
            axes[0, 1].set_title('Segmentation Loss', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

            # 先验损失
            axes[0, 2].plot(self.history['train_prior_loss'], label='Prior Loss',
                            color='orange', linewidth=2)
            axes[0, 2].set_title('Prior Mask Loss', fontweight='bold')
            axes[0, 2].legend()
            axes[0, 2].grid(alpha=0.3)

            # 对齐损失
            axes[1, 0].plot(self.history['train_align_loss'], label='Align Loss',
                            color='red', linewidth=2)
            axes[1, 0].set_title('Domain Alignment Loss', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

            # Dice
            axes[1, 1].plot(self.history['val_dice'], label='Val Dice',
                            color='green', linewidth=2)
            axes[1, 1].axhline(self.best_dice, color='r', linestyle='--',
                               label=f'Best: {self.best_dice:.4f}')
            axes[1, 1].set_title('Validation Dice', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

            # IoU
            axes[1, 2].plot(self.history['val_iou'], label='Val IoU',
                            color='purple', linewidth=2)
            axes[1, 2].set_title('Validation IoU', fontweight='bold')
            axes[1, 2].legend()
            axes[1, 2].grid(alpha=0.3)

            # 学习率
            axes[2, 0].plot(self.history['learning_rate'], color='brown', linewidth=2)
            axes[2, 0].set_title('Learning Rate', fontweight='bold')
            axes[2, 0].set_yscale('log')
            axes[2, 0].grid(alpha=0.3)

            # 时间
            axes[2, 1].plot(self.history['epoch_time'], color='navy', linewidth=2)
            axes[2, 1].axhline(np.mean(self.history['epoch_time']), color='r',
                               linestyle='--', label=f'Avg: {np.mean(self.history["epoch_time"]):.1f}s')
            axes[2, 1].set_title('Training Time', fontweight='bold')
            axes[2, 1].legend()
            axes[2, 1].grid(alpha=0.3)

            # Dice vs IoU
            axes[2, 2].plot(self.history['val_dice'], label='Dice', linewidth=2)
            axes[2, 2].plot(self.history['val_iou'], label='IoU', linewidth=2)
            axes[2, 2].set_title('Dice vs IoU', fontweight='bold')
            axes[2, 2].legend()
            axes[2, 2].grid(alpha=0.3)

            plt.tight_layout()
            save_path = self.config['checkpoint_dir'] / 'training_history.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ 训练曲线已保存: {save_path}")
            plt.show()

        except Exception as e:
            print(f"⚠️ 绘制失败: {e}")


# ==================== 9. 主训练函数 ====================
def train_cross_domain():
    """主训练函数"""

    print("\n" + "=" * 80)
    print("🚀 多模态跨域自适应UNet训练")
    print("=" * 80)

    config = {
        'data_root': r'D:\A基于UNet实现多模态跨域自适应\unet\Pytorch-UNet-master\data\mmwhs_processed',
        'n_classes': 2,
        'bilinear': True,
        'base_channels': 32,

        # 训练参数
        'epochs': 150,  # 🔥 增加到150
        'batch_size': 8,
        'lr': 7e-4,  # 🔥 略微降低
        'weight_decay': 6e-4,  # 🔥 增加正则化
        'accumulation_steps': 1,
        'use_amp': True,

        # 学习率调度
        'warmup_epochs': 10,  # 🔥 增加warmup
        'min_lr': 3e-7,  # 🔥 降低最小学习率

        # 早停
        'patience': 25,  # 🔥 增加patience
        'min_delta': 5e-5,

        # 数据加载
        'num_workers': 6,
        'pin_memory': True,
        'prefetch_factor': 3,

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'checkpoint_dir': Path('checkpoints_mmwhs_optimized') / datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 多模态跨域自适应UNet训练")
    print("=" * 80)
    print(f"\n【配置信息】")
    print(f"  数据路径: {config['data_root']}")
    print(f"  设备: {config['device']}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    print(f"\n【训练配置】")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Warmup Epochs: {config['warmup_epochs']}")
    print(f"  Patience: {config['patience']}")

    # 加载数据
    try:
        from mmwhs_dataset import create_mmwhs_loaders
        train_loader, val_loader = create_mmwhs_loaders(
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor']
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建模型
    # 创建模型
    try:
        # ✅ 直接使用本文件中定义的类
        model = MultiModalCrossDomainUNet(  # ← 改这里
            n_classes=config['n_classes'],
            bilinear=config['bilinear'],
            base_channels=config['base_channels']
        ).to(config['device'])

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n  【模型信息】")
        print(f"  参数量: {n_params:.2f}M")

    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 开始训练
    try:
        save_dir = config['checkpoint_dir']
        save_dir.mkdir(parents=True, exist_ok=True)

        trainer = CrossDomainTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            save_dir=save_dir  # 🔥 确保传入
        )
        trainer.train()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ GPU显存不足!")
            print("建议:")
            print("  1. batch_size 改为 4")
            print("  2. base_channels 改为 24")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

    # 🔥 删除这里的所有额外调用
    # train_cross_domain()  ← 删除这一行!!!

    print("\n✅ train_cross_domain() 函数执行完毕\n")


# ==================== 10. 程序入口 ====================

if __name__ == '__main__':
    import sys

    # 防止重复运行检查
    print("\n" + "🔔" * 40)
    print("📌 脚本启动检查")
    print(f"   当前脚本: {sys.argv[0]}")
    print(f"   Python版本: {sys.version}")
    print("🔔" * 40 + "\n")

    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("🎯 多模态跨域自适应UNet")
    print("=" * 80)
    print("\n【核心创新】")
    print("  1. 语义相似度自适应融合")
    print("     - CT和MRI特征动态权重分配")
    print("     - 解决特征错位和语义不一致")
    print("\n  2. 动态多模态先验掩码引导")
    print("     - CT结构先验 + MRI细节先验")
    print("     - 粗到精的分割引导")
    print("\n  3. 跨域自适应对齐")
    print("     - 对抗训练实现域不变特征")
    print("     - CT(源域) ↔ MRI(目标域)对齐")
    print("\n  4. 置信度引导的融合")
    print("     - 评估先验可靠性")
    print("     - 自适应调整融合策略")
    print("\n" + "=" * 80)

    # 🔥 只调用一次训练函数
    try:
        train_cross_domain()
        print("\n✅ 训练流程正常结束")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("\n" + "=" * 80)
        print("✅ 训练脚本执行完成!")
        print("=" * 80)

        # 🔥 强制退出，防止任何可能的重复
        print("\n🛑 即将退出...")
        sys.exit(0)