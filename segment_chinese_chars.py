#!/usr/bin/env python3
"""
SAM2 自动分割汉字脚本
从 pic1.jpg 中自动分割每个汉字，存入 segmented/ 目录
"""

import os
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt

def main():
    # 配置路径
    checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    image_path = "../../01-preResearch/pic1.jpg"
    output_dir = "../../segmented"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")

    print(f"使用设备: {device}")

    # 加载图像
    print(f"加载图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"图像尺寸: {image_rgb.shape}")

    # 构建 SAM2 模型
    print("加载 SAM2 模型...")
    sam2 = build_sam2(model_cfg, checkpoint, device=device)

    # 创建自动掩码生成器
    # 调整参数以更好地分割单个汉字
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,  # 增加采样点以更好地检测小对象
        pred_iou_thresh=0.8,  # 提高 IoU 阈值以获得更好的质量
        stability_score_thresh=0.92,  # 提高稳定性阈值
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # 最小区域面积，过滤太小的区域
    )

    # 生成掩码
    print("开始生成掩码...")
    masks = mask_generator.generate(image_rgb)
    print(f"生成了 {len(masks)} 个掩码")

    # 按区域面积排序（从大到小）
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # 保存每个分割区域
    saved_count = 0
    for idx, mask_data in enumerate(masks):
        mask = mask_data['segmentation']

        # 获取掩码的边界框
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # 添加一些边距
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)

        # 裁剪区域
        cropped = image[y_min:y_max, x_min:x_max].copy()

        # 创建掩码的裁剪版本
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # 将掩码应用到裁剪的图像（背景设为白色）
        white_bg = np.ones_like(cropped) * 255
        cropped_with_mask = np.where(cropped_mask[:, :, None], cropped, white_bg)

        # 保存分割结果
        output_path = os.path.join(output_dir, f"char_{idx:03d}.jpg")
        cv2.imwrite(output_path, cropped_with_mask)
        saved_count += 1

        # 打印进度
        if (idx + 1) % 10 == 0:
            print(f"已保存 {idx + 1} 个分割区域...")

    print(f"\n完成! 共保存了 {saved_count} 个分割区域到 {output_dir}")

    # 可视化前几个结果
    print("生成可视化预览...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(min(10, len(masks))):
        img_path = os.path.join(output_dir, f"char_{i:03d}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"Char {i}")
            axes[i].axis('off')

    preview_path = os.path.join(output_dir, "preview.png")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    print(f"预览图已保存到: {preview_path}")

    print("\n分割统计:")
    print(f"- 总分割数量: {len(masks)}")
    print(f"- 保存的区域: {saved_count}")
    areas = [m['area'] for m in masks]
    print(f"- 平均区域大小: {np.mean(areas):.0f} 像素")
    print(f"- 最大区域: {np.max(areas):.0f} 像素")
    print(f"- 最小区域: {np.min(areas):.0f} 像素")

if __name__ == "__main__":
    main()
