import os
from PIL import Image

# ----------- 配置部分 -----------
input_folder = './masks'  # 替换为你的文件夹路径，例如 'images'
output_suffix = '_flipped'          # 新文件后缀
supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # 支持的图像格式

# ----------- 主处理部分 -----------
for filename in os.listdir(input_folder):
    if filename.lower().endswith(supported_exts):
        input_path = os.path.join(input_folder, filename)

        # 打开图像并旋转180°
        image = Image.open(input_path)
        flipped_image = image.rotate(180)

        # 构造输出路径（添加后缀）
        name, ext = os.path.splitext(filename)
        output_name = f"{name}{output_suffix}{ext}"
        output_path = os.path.join(input_folder, output_name)

        # 保存新图像
        flipped_image.save(output_path)
        print(f"已保存: {output_path}")
