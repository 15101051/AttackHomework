"""
计算原始图片和对抗样本之间的SSIM值
原始图片目录: resources/lesson/images
对抗样本目录: results/lesson/images_autoattack_test
"""

import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def load_image_as_array(image_path):
    """加载图片并转换为numpy数组，格式为 (C, H, W)，值范围 [0, 1]"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0  # 归一化到 [0, 1]
    img_array = img_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return img_array


def calculate_ssim(original_path, adversarial_path):
    """计算两张图片之间的SSIM值"""
    original = load_image_as_array(original_path)
    adversarial = load_image_as_array(adversarial_path)
    
    # 计算SSIM，参考 TransferAttackAcc.py 中的方式
    data_range = original.max() - original.min()
    ssim_value = ssim(
        adversarial, 
        original, 
        data_range=data_range, 
        channel_axis=0
    )
    return ssim_value


def main():
    # 目录路径
    original_dir = './resources/lesson/images'
    adversarial_dir = './results/lesson/images_autoattack_test'
    
    # 获取所有图片文件名
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')], 
                           key=lambda x: int(x.split('.')[0]))
    
    print(f"原始图片目录: {original_dir}")
    print(f"对抗样本目录: {adversarial_dir}")
    print(f"图片数量: {len(original_files)}")
    print("-" * 60)
    
    ssim_values = []
    
    # 计算每张图片的SSIM
    for filename in tqdm(original_files, desc="计算SSIM"):
        original_path = os.path.join(original_dir, filename)
        adversarial_path = os.path.join(adversarial_dir, filename)
        
        if not os.path.exists(adversarial_path):
            print(f"警告: 对抗样本不存在 - {filename}")
            continue
        
        ssim_value = calculate_ssim(original_path, adversarial_path)
        ssim_values.append((filename, ssim_value))
    
    # 输出每张图片的SSIM值
    print("\n" + "=" * 60)
    print("每张图片的SSIM值:")
    print("=" * 60)
    print(f"{'图片名称':<15} {'SSIM值':<10}")
    print("-" * 60)
    
    for filename, ssim_val in ssim_values:
        print(f"{filename:<15} {ssim_val:.6f}")
    
    # 统计信息
    ssim_only = [s for _, s in ssim_values]
    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    print(f"平均 SSIM: {np.mean(ssim_only):.6f}")
    print(f"最小 SSIM: {np.min(ssim_only):.6f}")
    print(f"最大 SSIM: {np.max(ssim_only):.6f}")
    print(f"标准差:    {np.std(ssim_only):.6f}")
    
    # 找出SSIM最低的5张图片
    sorted_ssim = sorted(ssim_values, key=lambda x: x[1])
    print("\nSSIM最低的5张图片:")
    for filename, ssim_val in sorted_ssim[:5]:
        print(f"  {filename}: {ssim_val:.6f}")
    
    # 找出SSIM最高的5张图片
    print("\nSSIM最高的5张图片:")
    for filename, ssim_val in sorted_ssim[-5:]:
        print(f"  {filename}: {ssim_val:.6f}")


if __name__ == '__main__':
    main()

