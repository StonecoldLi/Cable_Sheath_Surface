import os
import json
import numpy as np
from PIL import Image
import cv2

def batch_json_to_mask(input_dir, output_dir):
    """
    批量转换JSON标注文件为Mask图片
    参数：
        input_dir: 包含JSON文件的输入目录
        output_dir: 保存Mask图片的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    
    # 计数器
    success = 0
    failed = []
    
    # 遍历处理
    for idx, json_file in enumerate(json_files, 1):
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(json_file)[0]}.png")
        
        print(f"Processing [{idx}/{len(json_files)}] {json_file}...", end=' ')
        
        try:
            # 读取JSON文件
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建空白mask
            mask = np.zeros((data['imageHeight'], data['imageWidth']), dtype=np.uint8)
            
            # 绘制所有多边形
            for shape in data['shapes']:
                if shape['shape_type'] == 'polygon':
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], color=1)
            
            # 保存结果
            Image.fromarray(mask * 255).save(output_path)
            success += 1
            print("✓")
            
        except Exception as e:
            failed.append(json_file)
            print(f"✗ (Error: {str(e)})")
    
    # 输出统计
    print("\nConversion Summary:")
    print(f"Success: {success}/{len(json_files)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f" - {f}")

# 使用示例
if __name__ == "__main__":
    input_folder = "./labels"  # 替换为你的输入目录
    output_folder = "./masks" # 替换为输出目录
    
    batch_json_to_mask(input_folder, output_folder)