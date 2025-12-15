"""
MMWHS数据集加载器
适配train_cross_domain.py的多模态训练
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import random
from scipy.ndimage import zoom

class MMWHSDataset(Dataset):
    """MMWHS配对CT-MR数据集"""

    def __init__(self, data_root, split='train', augment=False):
        """
        Args:
            data_root: 处理后的数据根目录
            split: 'train' 或 'val'
            augment: 是否进行数据增强
        """
        self.data_root = Path(data_root)
        self.split = split
        self.augment = augment

        # 数据路径（只使用训练集，因为测试集没有标注）
        self.ct_img_dir = self.data_root / 'train' / 'ct' / 'images'
        self.ct_mask_dir = self.data_root / 'train' / 'ct' / 'masks'
        self.mr_img_dir = self.data_root / 'train' / 'mr' / 'images'
        self.mr_mask_dir = self.data_root / 'train' / 'mr' / 'masks'

        # 获取文件列表
        self.ct_files = sorted(self.ct_img_dir.glob('*.png'))
        self.mr_files = sorted(self.mr_img_dir.glob('*.png'))

        # 确保数据存在
        if len(self.ct_files) == 0 or len(self.mr_files) == 0:
            raise ValueError(f"数据集为空！CT: {len(self.ct_files)}, MR: {len(self.mr_files)}")

        # 取较少的一方作为配对数量
        self.num_samples = min(len(self.ct_files), len(self.mr_files))

        # 划分训练集和验证集（80/20）
        if split == 'train':
            self.indices = list(range(0, int(0.8 * self.num_samples)))
        else:  # val
            self.indices = list(range(int(0.8 * self.num_samples), self.num_samples))

        print(f"MMWHS {split}集: {len(self.indices)} 个配对样本")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """返回格式：{'ct_image': ..., 'ct_mask': ..., 'mri_image': ..., 'mri_mask': ...,
                      'ct_prior': ..., 'mri_prior': ...}"""
        real_idx = self.indices[idx]

        # 加载CT
        ct_img_path = self.ct_files[real_idx]
        ct_mask_path = self.ct_mask_dir / ct_img_path.name
        ct_img = Image.open(ct_img_path).convert('L')
        ct_mask = Image.open(ct_mask_path).convert('L')

        # 加载MR
        mr_img_path = self.mr_files[real_idx]
        mr_mask_path = self.mr_mask_dir / mr_img_path.name
        mr_img = Image.open(mr_img_path).convert('L')
        mr_mask = Image.open(mr_mask_path).convert('L')

        # 转numpy
        ct_img = np.array(ct_img, dtype=np.float32) / 255.0
        mr_img = np.array(mr_img, dtype=np.float32) / 255.0
        ct_mask = np.array(ct_mask, dtype=np.float32) / 255.0
        mr_mask = np.array(mr_mask, dtype=np.float32) / 255.0

        # 🔥 生成弱先验（从图像生成，不是从mask）
        ct_prior = self._generate_weak_prior(ct_img)
        mri_prior = self._generate_weak_prior(mr_img)

        # 数据增强
        if self.augment:
            ct_img, ct_mask, mr_img, mr_mask, ct_prior, mri_prior = self._augment(
                ct_img, ct_mask, mr_img, mr_mask, ct_prior, mri_prior
            )

        # 转tensor
        ct_img = torch.from_numpy(ct_img.astype(np.float32)).unsqueeze(0)
        mr_img = torch.from_numpy(mr_img.astype(np.float32)).unsqueeze(0)

        ct_prior = torch.from_numpy(ct_prior.astype(np.float32)).unsqueeze(0)
        mri_prior = torch.from_numpy(mri_prior.astype(np.float32)).unsqueeze(0)

        # mask二值化
        ct_mask = (ct_mask > 0.5).astype(np.int64)
        mr_mask = (mr_mask > 0.5).astype(np.int64)
        ct_mask = torch.from_numpy(ct_mask).long()
        mr_mask = torch.from_numpy(mr_mask).long()

        return {
            'ct_image': ct_img,
            'ct_mask': ct_mask,
            'mri_image': mr_img,
            'mri_mask': mr_mask,
            'ct_prior': ct_prior,  # 🔥 新增
            'mri_prior': mri_prior  # 🔥 新增
        }

    def _generate_weak_prior(self, img):
        """多尺度先验融合（改进版）"""
        from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
        from skimage.filters import threshold_otsu, threshold_li

        priors = []

        # 🔥 方法1: Otsu自适应阈值
        try:
            t1 = threshold_otsu(img)
            b1 = (img > t1).astype(np.float32)
            # 形态学操作（更强）
            b1 = binary_opening(b1, iterations=3)
            b1 = binary_closing(b1, iterations=4)
            # 高斯平滑
            p1 = gaussian_filter(b1, sigma=6)
            priors.append(p1)
        except:
            pass

        # 🔥 方法2: Li阈值（对低对比度图像更鲁棒）
        try:
            t2 = threshold_li(img)
            b2 = (img > t2).astype(np.float32)
            b2 = binary_opening(b2, iterations=2)
            b2 = binary_closing(b2, iterations=3)
            p2 = gaussian_filter(b2, sigma=8)
            priors.append(p2)
        except:
            pass

        # 🔥 方法3: 百分位数（鲁棒的备选方案）
        t3 = np.percentile(img, 65)  # 65分位数
        b3 = (img > t3).astype(np.float32)
        b3 = binary_opening(b3, iterations=2)
        b3 = binary_closing(b3, iterations=3)
        p3 = gaussian_filter(b3, sigma=7)
        priors.append(p3)

        # 🔥 融合多个先验（加权平均，优先使用Otsu）
        if len(priors) == 3:
            # Otsu权重0.5, Li权重0.3, 百分位权重0.2
            prior = 0.5 * priors[0] + 0.3 * priors[1] + 0.2 * priors[2]
        elif len(priors) == 2:
            prior = 0.6 * priors[0] + 0.4 * priors[1]
        else:
            prior = priors[0]

        # 归一化
        if prior.max() > 0:
            prior = prior / prior.max()

        # 🔥 防止过度平滑（保留细节）
        prior = np.clip(prior, 0.0, 1.0)

        return prior

    def _augment(self, ct_img, ct_mask, mri_img, mri_mask, ct_prior, mri_prior):
        """增强的数据增强"""

        # 1. 随机水平翻转（60%概率）
        if random.random() > 0.4:
            ct_img = np.fliplr(ct_img).copy()
            ct_mask = np.fliplr(ct_mask).copy()
            ct_prior = np.fliplr(ct_prior).copy()
            mri_img = np.fliplr(mri_img).copy()
            mri_mask = np.fliplr(mri_mask).copy()
            mri_prior = np.fliplr(mri_prior).copy()

        # 2. 随机垂直翻转（60%概率）
        if random.random() > 0.4:
            ct_img = np.flipud(ct_img).copy()
            ct_mask = np.flipud(ct_mask).copy()
            ct_prior = np.flipud(ct_prior).copy()
            mri_img = np.flipud(mri_img).copy()
            mri_mask = np.flipud(mri_mask).copy()
            mri_prior = np.flipud(mri_prior).copy()

        # 3. 随机旋转（50%概率）
        if random.random() > 0.5:
            k = random.randint(1, 3)
            ct_img = np.rot90(ct_img, k).copy()
            ct_mask = np.rot90(ct_mask, k).copy()
            ct_prior = np.rot90(ct_prior, k).copy()
            mri_img = np.rot90(mri_img, k).copy()
            mri_mask = np.rot90(mri_mask, k).copy()
            mri_prior = np.rot90(mri_prior, k).copy()

        # 🔥 4. 弹性变形（新增，30%概率）
        if random.random() > 0.7:
            from scipy.ndimage import map_coordinates, gaussian_filter

            def elastic_transform(image, alpha=30, sigma=5):
                shape = image.shape
                dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
                dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

                x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

                return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

            ct_img = elastic_transform(ct_img)
            ct_mask = elastic_transform(ct_mask)
            ct_prior = elastic_transform(ct_prior)
            mri_img = elastic_transform(mri_img)
            mri_mask = elastic_transform(mri_mask)
            mri_prior = elastic_transform(mri_prior)

        # 5. 高斯噪声（40%概率，降低强度）
        if random.random() > 0.6:
            noise_std = random.uniform(0.005, 0.015)  # 🔥 降低噪声
            ct_img = np.clip(ct_img + np.random.randn(*ct_img.shape) * noise_std, 0, 1)
            mri_img = np.clip(mri_img + np.random.randn(*mri_img.shape) * noise_std, 0, 1)

        # 6. 对比度调整（30%概率）
        if random.random() > 0.7:
            gamma = random.uniform(0.85, 1.15)  # 🔥 更温和的gamma
            ct_img = np.power(ct_img, gamma)
            mri_img = np.power(mri_img, gamma)

        # 🔥 7. 随机亮度调整（新增，30%概率）
        if random.random() > 0.7:
            brightness = random.uniform(0.9, 1.1)
            ct_img = np.clip(ct_img * brightness, 0, 1)
            mri_img = np.clip(mri_img * brightness, 0, 1)

        return ct_img, ct_mask, mri_img, mri_mask, ct_prior, mri_prior




def create_mmwhs_loaders(data_root, batch_size=4, num_workers=4, pin_memory=True, prefetch_factor=2):
    """
    创建MMWHS数据加载器

    返回格式与原代码一致：train_loader, val_loader
    """

    # 训练集（带数据增强）
    train_dataset = MMWHSDataset(
        data_root=data_root,
        split='train',
        augment=True
    )

    # 验证集（不增强）
    val_dataset = MMWHSDataset(
        data_root=data_root,
        split='val',
        augment=False
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"\n数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  批次大小: {batch_size}")

    return train_loader, val_loader


# 测试代码
if __name__ == '__main__':
    data_root = r'D:\A基于UNet实现多模态跨域自适应\unet\Pytorch-UNet-master\data\mmwhs_processed'

    train_loader, val_loader = create_mmwhs_loaders(data_root, batch_size=2)

    # ✅ 修改测试部分
    print("\n测试数据加载:")
    for batch in train_loader:
        ct_img = batch['ct_image']
        ct_mask = batch['ct_mask']
        mri_img = batch['mri_image']
        mri_mask = batch['mri_mask']

        print(f"  CT图像: {ct_img.shape}, 范围: [{ct_img.min():.3f}, {ct_img.max():.3f}]")
        print(f"  CT标注: {ct_mask.shape}, 类别: {torch.unique(ct_mask)}")
        print(f"  MR图像: {mri_img.shape}, 范围: [{mri_img.min():.3f}, {mri_img.max():.3f}]")
        print(f"  MR标注: {mri_mask.shape}, 类别: {torch.unique(mri_mask)}")
        break