#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像重命名脚本
将app/images路径下的图像文件重命名为left_001.png格式，并复制到src/data/leftImg目录下
"""

import os
import shutil
import glob
from pathlib import Path

def rename_and_copy_images():
    """
    重命名并复制图像文件
    """
    # 源目录和目标目录
    source_dir = "/home/lz/camera_lidar_cail/src/data/leftImg_pre"
    target_dir = "/home/lz/camera_lidar_cail/src/data/leftImg"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有PNG文件
    image_files = glob.glob(os.path.join(source_dir, "*.png"))
    
    # 按文件名排序
    image_files.sort()
    
    print(f"找到 {len(image_files)} 个PNG文件")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print("-" * 50)
    
    # 重命名并复制文件
    for i, image_file in enumerate(image_files, 1):
        # 生成新的文件名
        new_filename = f"left_{i:03d}.png"
        target_path = os.path.join(target_dir, new_filename)
        
        # 复制文件
        try:
            shutil.copy2(image_file, target_path)
            print(f"复制: {os.path.basename(image_file)} -> {new_filename}")
        except Exception as e:
            print(f"错误: 复制 {image_file} 失败 - {e}")
    
    print("-" * 50)
    print(f"完成！已处理 {len(image_files)} 个文件")
    print(f"文件已保存到: {target_dir}")

def create_names_file():
    """
    创建names.txt文件，包含图像和lidar文件的映射关系
    """
    names_file = "/home/lz/camera_lidar_cail/src/data/names.txt"
    
    # 获取图像文件数量
    target_dir = "/home/lz/camera_lidar_cail/src/data/leftImg"
    image_files = glob.glob(os.path.join(target_dir, "left_*.png"))
    num_images = len(image_files)
    
    print(f"\n创建names.txt文件，包含 {num_images} 个图像映射...")
    
    with open(names_file, 'w', encoding='utf-8') as f:
        f.write("# 使用的时候需要把这几行注释都删除，只保留数据\n")
        f.write("# 格式：lidar文件名 图像编号\n")
        f.write("# 每行格式：lidar_xxx 001\n")
        f.write("# 其中lidar_xxx是lidar文件名（不包含.txt扩展名），001是图像编号\n\n")
        
        for i in range(1, num_images + 1):
            lidar_name = f"lidar_{i:02d}"
            image_num = f"{i:03d}"
            f.write(f"{lidar_name} {image_num}\n")
    
    print(f"names.txt文件已创建: {names_file}")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("图像重命名和复制脚本")
    print("=" * 60)
    
    # 检查源目录是否存在
    source_dir = "/home/lz/camera_lidar_cail/app/images"
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在 - {source_dir}")
        return
    
    # 重命名并复制图像
    rename_and_copy_images()
    
    # 创建names.txt文件
    create_names_file()
    
    print("\n" + "=" * 60)
    print("所有操作完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

