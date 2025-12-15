"""
MMWHS数据集完整处理流程（修复版）
包括：数据预处理、质量检测、配对验证
关键修复：
1. 统计信息保存移到循环外
2. 降低质量筛选阈值
3. 支持测试集无标注
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import json


class MMWHSPreprocessor:
    """MMWHS数据集预处理器"""

    def __init__(self, data_root, output_root):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)

        # 数据路径
        self.ct_train_dir = self.data_root / 'ct_train'
        self.ct_test_dir = self.data_root / 'ct_test'
        self.mr_train_dir = self.data_root / 'mr_train'
        self.mr_test_dir = self.data_root / 'mr_test'

        # 输出路径
        self.output_root.mkdir(parents=True, exist_ok=True)

        # 统计信息
        self.stats = {
            'train': {
                'ct': {'volumes': [], 'slices': [], 'quality': [], 'spacing': [], 'dimensions': []},
                'mr': {'volumes': [], 'slices': [], 'quality': [], 'spacing': [], 'dimensions': []}
            },
            'test': {
                'ct': {'volumes': [], 'slices': [], 'quality': [], 'spacing': [], 'dimensions': []},
                'mr': {'volumes': [], 'slices': [], 'quality': [], 'spacing': [], 'dimensions': []}
            }
        }

    def load_nifti(self, file_path):
        """加载NIfTI文件"""
        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
            affine = img.affine
            header = img.header
            spacing = header.get_zooms()
            return data, affine, header, spacing
        except Exception as e:
            print(f"❌ 加载失败: {file_path}")
            print(f"   错误: {e}")
            return None, None, None, None

    def normalize_intensity(self, img, modality='ct'):
        """标准化图像强度"""
        if modality == 'ct':
            # CT: 心脏窗
            window_center = 40
            window_width = 400
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2

            img = np.clip(img, img_min, img_max)
            img = (img - img_min) / (img_max - img_min) * 255

        else:  # MR
            # 百分位数归一化
            valid_pixels = img[img > 0]
            if len(valid_pixels) > 0:
                p1 = np.percentile(valid_pixels, 1)
                p99 = np.percentile(valid_pixels, 99)
                img = np.clip(img, p1, p99)
                img = (img - p1) / (p99 - p1 + 1e-8) * 255
            else:
                img = np.zeros_like(img)

        return img.astype(np.uint8)

    def check_slice_quality(self, slice_img, mask_slice=None):
        """
        检查切片质量（宽松版本）
        """
        quality_score = 1.0
        issues = []

        # 1. 检查是否全黑
        if np.max(slice_img) == 0:
            return 0.0, ['全黑切片']

        # 2. 检查对比度（降低惩罚）
        contrast = np.std(slice_img)
        if contrast < 5:  # ← 从10降到5
            quality_score *= 0.7  # ← 从0.5改到0.7
            issues.append(f'低对比度({contrast:.1f})')

        # 3. 检查是否包含目标（降低惩罚）
        if mask_slice is not None:
            target_pixels = np.sum(mask_slice > 0)
            if target_pixels == 0:
                quality_score *= 0.5  # ← 从0.3改到0.5
                issues.append('无目标')
            elif target_pixels < 50:  # ← 从100降到50
                quality_score *= 0.8  # ← 从0.7改到0.8
                issues.append(f'目标小({target_pixels}px)')

        # 4. 信噪比检查（降低惩罚）
        foreground = slice_img[slice_img > np.percentile(slice_img, 10)]
        if len(foreground) > 0:
            signal = np.mean(foreground)
            noise = np.std(foreground)
            if noise > 0:
                snr = signal / noise
                if snr < 1.5:  # ← 从2降到1.5
                    quality_score *= 0.8  # ← 从0.7改到0.8
                    issues.append(f'低SNR({snr:.1f})')

        return min(quality_score, 1.0), issues

    def process_volume(self, img_path, mask_path, output_dir, modality, volume_id):
        """
        处理单个3D volume（修复版）
        ✅ 关键修复：统计信息保存移到循环外
        """
        print(f"\n处理 {modality.upper()} Volume {volume_id}: {img_path.name}")

        # 加载图像
        img_data, img_affine, img_header, img_spacing = self.load_nifti(img_path)
        if img_data is None:
            return 0

        # 加载标注
        mask_data = None
        if mask_path and mask_path.exists():
            mask_data, _, _, _ = self.load_nifti(mask_path)
            if mask_data is not None:
                unique_labels = np.unique(mask_data)
                print(f"  ✅ 标注类别: {unique_labels}")
        else:
            print(f"  ⚠️  未找到标注文件（测试集）")

        print(f"  图像形状: {img_data.shape}")
        print(f"  体素间距: {img_spacing}")

        # 创建输出目录
        img_output_dir = output_dir / modality / 'images'
        mask_output_dir = output_dir / modality / 'masks'
        img_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        # 逐切片处理
        num_slices = img_data.shape[2]
        valid_slices = 0
        quality_scores = []

        for slice_idx in tqdm(range(num_slices), desc=f"  处理切片", leave=False):
            # 提取切片
            img_slice = img_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx] if mask_data is not None else None

            # 检查质量
            quality, issues = self.check_slice_quality(img_slice, mask_slice)
            quality_scores.append(quality)

            # ✅ 降低质量阈值到0.2（而不是0.3）
            if quality < 0.2:
                continue

            # 标准化
            img_slice_norm = self.normalize_intensity(img_slice, modality)

            # 调整大小
            img_pil = Image.fromarray(img_slice_norm)
            img_resized = img_pil.resize((256, 256), Image.BILINEAR)

            # 保存图像
            save_name = f"{modality}_vol{volume_id:03d}_slice{slice_idx:03d}.png"
            img_resized.save(img_output_dir / save_name)

            # 保存标注
            if mask_slice is not None:
                mask_binary = (mask_slice > 0).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_binary)
                mask_resized = mask_pil.resize((256, 256), Image.NEAREST)
                mask_resized.save(mask_output_dir / save_name)
            # 测试集没有标注，不保存mask

            valid_slices += 1

        # ✅✅✅ 关键修复：统计信息保存要在循环外面！
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        split = 'test' if 'test' in str(output_dir) else 'train'

        self.stats[split][modality]['volumes'].append(volume_id)
        self.stats[split][modality]['slices'].append(valid_slices)
        self.stats[split][modality]['quality'].append(avg_quality)
        self.stats[split][modality]['spacing'].append(img_spacing)
        self.stats[split][modality]['dimensions'].append(img_data.shape)

        print(f"  ✅ 有效切片: {valid_slices}/{num_slices} (质量分数: {avg_quality:.3f})")

        return valid_slices

    def process_split(self, split='train'):
        """
        处理指定的数据集分割
        ✅ 支持测试集无标注
        """

        if split == 'train':
            ct_dir = self.ct_train_dir
            mr_dir = self.mr_train_dir
        else:
            ct_dir = self.ct_test_dir
            mr_dir = self.mr_test_dir

        print(f"\n📂 数据目录：")
        print(f"  CT: {ct_dir} (存在: {ct_dir.exists()})")
        print(f"  MR: {mr_dir} (存在: {mr_dir.exists()})")

        ct_files = {'images': [], 'labels': []}
        mr_files = {'images': [], 'labels': []}

        # 扫描CT
        if ct_dir.exists():
            print(f"\n🔍 扫描CT {split}集...")
            all_ct_files = list(ct_dir.glob('*.nii*'))
            print(f"  CT目录下共有 {len(all_ct_files)} 个.nii文件")

            for file in all_ct_files:
                if 'image' in file.name:
                    ct_files['images'].append(file)
                elif 'label' in file.name:
                    ct_files['labels'].append(file)

            ct_files['images'] = sorted(ct_files['images'])
            ct_files['labels'] = sorted(ct_files['labels'])

            print(f"  ✅ CT图像: {len(ct_files['images'])} 个")
            print(f"  ✅ CT标注: {len(ct_files['labels'])} 个")

            if len(ct_files['images']) > 0:
                print(f"  示例文件: {ct_files['images'][0].name}")

        # 扫描MR
        if mr_dir.exists():
            print(f"\n🔍 扫描MR {split}集...")
            all_mr_files = list(mr_dir.glob('*.nii*'))
            print(f"  MR目录下共有 {len(all_mr_files)} 个.nii文件")

            for file in all_mr_files:
                if 'image' in file.name:
                    mr_files['images'].append(file)
                elif 'label' in file.name:
                    mr_files['labels'].append(file)

            mr_files['images'] = sorted(mr_files['images'])
            mr_files['labels'] = sorted(mr_files['labels'])

            print(f"  ✅ MR图像: {len(mr_files['images'])} 个")
            print(f"  ✅ MR标注: {len(mr_files['labels'])} 个")

            if len(mr_files['images']) > 0:
                print(f"  示例文件: {mr_files['images'][0].name}")

        # 如果没找到任何文件，返回
        if len(ct_files['images']) == 0 and len(mr_files['images']) == 0:
            print(f"\n⚠️  {split}集中未找到任何数据文件，跳过...")
            return

        # 处理CT
        if len(ct_files['images']) > 0:
            print(f"\n{'=' * 80}")
            print(f"【处理CT {split.upper()}集】")
            print(f"{'=' * 80}")

            for idx, img_path in enumerate(ct_files['images'], 1):
                label_name = img_path.name.replace('_image', '_label')
                mask_path = img_path.parent / label_name
                if not mask_path.exists():
                    mask_path = None

                self.process_volume(
                    img_path=img_path,
                    mask_path=mask_path,
                    output_dir=self.output_root / split,
                    modality='ct',
                    volume_id=idx
                )

        # 处理MR
        if len(mr_files['images']) > 0:
            print(f"\n{'=' * 80}")
            print(f"【处理MR {split.upper()}集】")
            print(f"{'=' * 80}")

            for idx, img_path in enumerate(mr_files['images'], 1):
                label_name = img_path.name.replace('_image', '_label')
                mask_path = img_path.parent / label_name
                if not mask_path.exists():
                    mask_path = None

                self.process_volume(
                    img_path=img_path,
                    mask_path=mask_path,
                    output_dir=self.output_root / split,
                    modality='mr',
                    volume_id=idx
                )

    def process_all(self):
        """处理所有数据"""
        print("=" * 80)
        print("🚀 MMWHS数据集预处理")
        print("=" * 80)

        self.process_split('train')
        self.process_split('test')

        self.generate_reports()

    def generate_reports(self):
        """生成报告"""
        print("\n" + "="*80)
        print("📊 生成数据质量报告")
        print("="*80)

        self.generate_text_report()
        self.save_stats_json()

        print("\n" + "="*80)
        print(f"✅ 所有处理完成！")
        print(f"   数据保存在: {self.output_root}")
        print(f"   报告文件:")
        print(f"     - quality_report.txt")
        print(f"     - stats.json")
        print("="*80)

    def generate_text_report(self):
        """生成文本报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MMWHS数据集处理报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        # 训练集统计
        report_lines.append("【训练集统计】")
        report_lines.append("")

        if len(self.stats['train']['ct']['volumes']) > 0:
            report_lines.append("CT训练数据：")
            report_lines.append(f"  Volume数量: {len(self.stats['train']['ct']['volumes'])}")
            report_lines.append(f"  总切片数: {sum(self.stats['train']['ct']['slices'])}")
            report_lines.append(f"  平均每个volume: {np.mean(self.stats['train']['ct']['slices']):.1f} 切片")
            report_lines.append(f"  平均质量分数: {np.mean(self.stats['train']['ct']['quality']):.3f}")
            report_lines.append("")

        if len(self.stats['train']['mr']['volumes']) > 0:
            report_lines.append("MR训练数据：")
            report_lines.append(f"  Volume数量: {len(self.stats['train']['mr']['volumes'])}")
            report_lines.append(f"  总切片数: {sum(self.stats['train']['mr']['slices'])}")
            report_lines.append(f"  平均每个volume: {np.mean(self.stats['train']['mr']['slices']):.1f} 切片")
            report_lines.append(f"  平均质量分数: {np.mean(self.stats['train']['mr']['quality']):.3f}")
            report_lines.append("")

        # 测试集统计
        report_lines.append("【测试集统计】")
        report_lines.append("")

        if len(self.stats['test']['ct']['volumes']) > 0:
            report_lines.append("CT测试数据：")
            report_lines.append(f"  Volume数量: {len(self.stats['test']['ct']['volumes'])}")
            report_lines.append(f"  总切片数: {sum(self.stats['test']['ct']['slices'])}")
            report_lines.append(f"  平均每个volume: {np.mean(self.stats['test']['ct']['slices']):.1f} 切片")
            report_lines.append(f"  平均质量分数: {np.mean(self.stats['test']['ct']['quality']):.3f}")
            report_lines.append("")

        if len(self.stats['test']['mr']['volumes']) > 0:
            report_lines.append("MR测试数据：")
            report_lines.append(f"  Volume数量: {len(self.stats['test']['mr']['volumes'])}")
            report_lines.append(f"  总切片数: {sum(self.stats['test']['mr']['slices'])}")
            report_lines.append(f"  平均每个volume: {np.mean(self.stats['test']['mr']['slices']):.1f} 切片")
            report_lines.append(f"  平均质量分数: {np.mean(self.stats['test']['mr']['quality']):.3f}")
            report_lines.append("")

        # 总体统计
        report_lines.append("【总体统计】")
        total_train_ct = sum(self.stats['train']['ct']['slices']) if self.stats['train']['ct']['slices'] else 0
        total_train_mr = sum(self.stats['train']['mr']['slices']) if self.stats['train']['mr']['slices'] else 0
        total_test_ct = sum(self.stats['test']['ct']['slices']) if self.stats['test']['ct']['slices'] else 0
        total_test_mr = sum(self.stats['test']['mr']['slices']) if self.stats['test']['mr']['slices'] else 0

        report_lines.append(f"  训练集总切片: {total_train_ct + total_train_mr}")
        report_lines.append(f"    - CT: {total_train_ct}")
        report_lines.append(f"    - MR: {total_train_mr}")
        report_lines.append(f"  测试集总切片: {total_test_ct + total_test_mr}")
        report_lines.append(f"    - CT: {total_test_ct}")
        report_lines.append(f"    - MR: {total_test_mr}")
        report_lines.append(f"  总计切片数: {total_train_ct + total_train_mr + total_test_ct + total_test_mr}")
        report_lines.append("")

        report_lines.append("=" * 80)

        # 保存报告
        report_text = "\n".join(report_lines)
        with open(self.output_root / 'quality_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)

    def save_stats_json(self):
        """保存统计信息到JSON（修复版）"""

        def convert_to_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # 转换统计信息
        stats_serializable = {}

        for split in ['train', 'test']:
            stats_serializable[split] = {}

            for modality in ['ct', 'mr']:
                stats_serializable[split][modality] = {
                    'volumes': [int(v) for v in self.stats[split][modality]['volumes']],
                    'slices': [int(s) for s in self.stats[split][modality]['slices']],
                    'quality': [float(q) for q in self.stats[split][modality]['quality']],
                    'spacing': [[float(x) for x in s] for s in self.stats[split][modality]['spacing']],
                    'dimensions': [[int(x) for x in d] for d in self.stats[split][modality]['dimensions']]
                }

        # 保存到文件
        with open(self.output_root / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats_serializable, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 统计信息已保存到: {self.output_root / 'stats.json'}")


# 主函数
if __name__ == '__main__':
    data_root = r'D:\A基于UNet实现多模态跨域自适应\unet\Pytorch-UNet-master\data\mmwhs'
    output_root = r'D:\A基于UNet实现多模态跨域自适应\unet\Pytorch-UNet-master\data\mmwhs_processed'

    processor = MMWHSPreprocessor(data_root, output_root)
    processor.process_all()

    print("\n✅ 预处理完成！")