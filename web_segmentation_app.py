#!/usr/bin/env python3
"""
SAM2 交互式汉字分割 Web 应用 - 升级版
支持框选自动分割、缩放和预览
"""

import os
import io
import base64
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__)

# 全局变量
predictor = None
current_image = None
current_image_rgb = None
segmentation_counter = 0
saved_segments = []  # 存储已保存的分割图像

# 配置
CHECKPOINT = "./checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMAGE_PATH = "../../01-preResearch/pic1.jpg"
OUTPUT_DIR = "../../segmented"
CONFIG_FILE = "adjustment_config.json"  # 配置文件路径

def init_sam2():
    """初始化 SAM2 模型"""
    global predictor, current_image, current_image_rgb

    # 选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"使用设备: {device}")

    # 加载模型
    print("加载 SAM2 模型...")
    sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2)

    # 加载图像
    print(f"加载图像: {IMAGE_PATH}")
    current_image = cv2.imread(IMAGE_PATH)
    current_image_rgb = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    # 设置图像
    predictor.set_image(current_image_rgb)
    print("初始化完成!")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def image_to_base64(image):
    """将 numpy 图像转换为 base64"""
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def load_adjustment_config():
    """加载图像调整配置"""
    default_config = {
        'brightness': 100,
        'contrast': 100,
        'saturation': 100,
        'invert_colors': False,
        'curve_points': [{'x': 0, 'y': 0}, {'x': 255, 'y': 255}]
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置失败: {e}")
            return default_config
    return default_config

def save_adjustment_config(config):
    """保存图像调整配置"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/get_image')
def get_image():
    """获取当前图像"""
    if current_image_rgb is None:
        return jsonify({'error': 'No image loaded'}), 400

    img_base64 = image_to_base64(current_image_rgb)
    return jsonify({
        'image': img_base64,
        'width': current_image_rgb.shape[1],
        'height': current_image_rgb.shape[0]
    })

def apply_adjustments(image, brightness, contrast, saturation, curve_lut=None, invert_colors=False):
    """应用图像调整参数"""
    # 转换为 float32 进行处理
    adjusted = image.astype(np.float32)

    # 应用亮度 (brightness: 0-200, 100 为原始)
    brightness_factor = brightness / 100.0
    adjusted = adjusted * brightness_factor

    # 应用对比度 (contrast: 0-200, 100 为原始)
    contrast_factor = contrast / 100.0
    mean = np.mean(adjusted, axis=(0, 1), keepdims=True)
    adjusted = (adjusted - mean) * contrast_factor + mean

    # 应用饱和度 (saturation: 0-200, 100 为原始)
    if saturation != 100:
        saturation_factor = saturation / 100.0
        # 转换到 HSV 色彩空间
        adjusted_uint8 = np.clip(adjusted, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(adjusted_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 裁剪到有效范围并转换回 uint8
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    # 应用曲线调整
    if curve_lut is not None and len(curve_lut) == 256:
        lut = np.array(curve_lut, dtype=np.uint8)
        adjusted = cv2.LUT(adjusted, lut)

    # 应用反色
    if invert_colors:
        adjusted = 255 - adjusted

    return adjusted

@app.route('/predict_box', methods=['POST'])
def predict_box():
    """根据框选区域预测分割掩码"""
    global segmentation_counter, saved_segments

    data = request.json
    box = np.array(data['box'])  # [x1, y1, x2, y2]

    # 获取图像调整参数（默认值为 100），转换为整数
    brightness = int(data.get('brightness', 100))
    contrast = int(data.get('contrast', 100))
    saturation = int(data.get('saturation', 100))
    curve_lut = data.get('curve_lut', None)  # 获取曲线查找表
    invert_colors = data.get('invert_colors', False)  # 获取反色参数

    # 使用框作为输入，启用多掩码输出以获得更好的分割
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],  # 添加批次维度
        multimask_output=True,  # 改为 True 获取多个候选掩码
    )

    # 选择得分最高的掩码
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    score = float(scores[best_idx])

    # 形态学操作优化掩码边缘
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = mask.astype(bool)

    # 自动保存分割结果
    y_indices, x_indices = np.where(mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # 添加较小的边距以获得更紧凑的结果
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(current_image.shape[1], x_max + padding)
        y_max = min(current_image.shape[0], y_max + padding)

        # 应用图像调整参数
        adjusted_image = apply_adjustments(current_image, brightness, contrast, saturation, curve_lut, invert_colors)

        # 裁剪调整后的图像
        cropped = adjusted_image[y_min:y_max, x_min:x_max].copy()
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # 应用掩码（白色背景）
        white_bg = np.ones_like(cropped) * 255
        cropped_with_mask = np.where(cropped_mask[:, :, None], cropped, white_bg)

        # 保存
        output_path = os.path.join(OUTPUT_DIR, f"char_{segmentation_counter:03d}.jpg")
        cv2.imwrite(output_path, cropped_with_mask)

        # 转换为 RGB 并生成 base64
        cropped_rgb = cv2.cvtColor(cropped_with_mask, cv2.COLOR_BGR2RGB)
        segment_base64 = image_to_base64(cropped_rgb)

        # 添加到已保存列表
        saved_segments.append({
            'id': segmentation_counter,
            'image': segment_base64,
            'path': output_path
        })

        segmentation_counter += 1

        return jsonify({
            'success': True,
            'score': score,
            'segment_image': segment_base64,
            'count': segmentation_counter,
            'path': output_path
        })
    else:
        return jsonify({'error': 'Empty mask'}), 400

@app.route('/get_saved_segments')
def get_saved_segments():
    """获取所有已保存的分割图像"""
    return jsonify({
        'segments': saved_segments,
        'count': len(saved_segments)
    })

@app.route('/delete_segment/<int:segment_id>', methods=['DELETE'])
def delete_segment(segment_id):
    """删除指定的分割图像"""
    global saved_segments

    for i, seg in enumerate(saved_segments):
        if seg['id'] == segment_id:
            # 删除文件
            if os.path.exists(seg['path']):
                os.remove(seg['path'])
            # 从列表中移除
            saved_segments.pop(i)
            return jsonify({'success': True})

    return jsonify({'error': 'Segment not found'}), 404

@app.route('/get_config')
def get_config():
    """获取图像调整配置"""
    config = load_adjustment_config()
    return jsonify(config)

@app.route('/save_config', methods=['POST'])
def save_config():
    """保存图像调整配置"""
    try:
        config = request.json
        if save_adjustment_config(config):
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Failed to save config'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("初始化 SAM2...")
    init_sam2()
    print("\n启动 Web 服务器...")
    print("请在浏览器中打开: http://localhost:5001")
    print("框选汉字进行自动分割")
    app.run(debug=False, host='0.0.0.0', port=5001)
