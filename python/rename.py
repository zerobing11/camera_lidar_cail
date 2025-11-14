#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像和雷达文件重命名脚本
从../data/output/img和../data/output/lidar文件夹中的文件按顺序重命名输出到../data/img和../data/lidar文件夹
同时生成../data/names.txt文件
"""

import os
import shutil
import glob

def rename_and_copy_images():
    """
    重命名并复制图像文件
    从../data/output/img复制到../data/img，格式为img_001.png
    """
    # 获取当前脚本目录的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    source_dir = os.path.join(parent_dir, "data", "output", "img")
    target_dir = os.path.join(parent_dir, "data", "img")
    
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
    copied_files = []
    for i, image_file in enumerate(image_files, 1):
        # 生成新的文件名
        new_filename = f"img_{i:03d}.png"
        target_path = os.path.join(target_dir, new_filename)
        
        # 复制文件
        try:
            shutil.copy2(image_file, target_path)
            copied_files.append(new_filename)
            print(f"复制: {os.path.basename(image_file)} -> {new_filename}")
        except Exception as e:
            print(f"错误: 复制 {image_file} 失败 - {e}")
    
    print("-" * 50)
    print(f"完成！已处理 {len(copied_files)} 个图像文件")
    return len(copied_files)

def rename_and_copy_lidar():
    """
    重命名并复制雷达文件
    从../data/output/lidar复制到../data/lidar，格式为lidar_001.ply
    """
    # 获取当前脚本目录的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    source_dir = os.path.join(parent_dir, "data", "output", "lidar")
    target_dir = os.path.join(parent_dir, "data", "lidar")
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有文件（支持.ply和.txt格式）
    lidar_files = []
    lidar_files.extend(glob.glob(os.path.join(source_dir, "*.ply")))
    lidar_files.extend(glob.glob(os.path.join(source_dir, "*.txt")))
    
    # 按文件名排序
    lidar_files.sort()
    
    print(f"找到 {len(lidar_files)} 个雷达文件")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print("-" * 50)
    
    # 重命名并复制文件
    copied_files = []
    for i, lidar_file in enumerate(lidar_files, 1):
        # 生成新的文件名（保持.ply格式）
        new_filename = f"lidar_{i:03d}.ply"
        target_path = os.path.join(target_dir, new_filename)
        
        # 复制文件
        try:
            shutil.copy2(lidar_file, target_path)
            copied_files.append(new_filename)
            print(f"复制: {os.path.basename(lidar_file)} -> {new_filename}")
        except Exception as e:
            print(f"错误: 复制 {lidar_file} 失败 - {e}")
    
    print("-" * 50)
    print(f"完成！已处理 {len(copied_files)} 个雷达文件")
    return len(copied_files)

def create_names_file(num_files):
    """
    创建names.txt文件，包含雷达和图像文件的映射关系
    格式：lidar_001 img_001
    """
    # 获取当前脚本目录的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    names_file = os.path.join(parent_dir, "data", "names.txt")
    
    print(f"\n创建names.txt文件，包含 {num_files} 个文件映射...")
    
    with open(names_file, 'w', encoding='utf-8') as f:
        for i in range(1, num_files + 1):
            lidar_name = f"lidar_{i:03d}"
            img_name = f"img_{i:03d}"
            f.write(f"{lidar_name} {img_name}\n")
    
    print(f"names.txt文件已创建: {names_file}")

def verify_directories():
    """
    验证源目录是否存在
    """
    # 获取当前脚本目录的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    img_pre_dir = os.path.join(parent_dir, "data", "output", "img")
    lidar_pre_dir = os.path.join(parent_dir, "data", "output", "lidar")
    
    errors = []
    
    if not os.path.exists(img_pre_dir):
        errors.append(f"图像源目录不存在: {img_pre_dir}")
    
    if not os.path.exists(lidar_pre_dir):
        errors.append(f"雷达源目录不存在: {lidar_pre_dir}")
    
    # 检查是否有文件
    if os.path.exists(img_pre_dir):
        img_files = glob.glob(os.path.join(img_pre_dir, "*.png"))
        if len(img_files) == 0:
            errors.append(f"图像源目录中没有PNG文件: {img_pre_dir}")
    
    if os.path.exists(lidar_pre_dir):
        lidar_files = []
        lidar_files.extend(glob.glob(os.path.join(lidar_pre_dir, "*.ply")))
        lidar_files.extend(glob.glob(os.path.join(lidar_pre_dir, "*.txt")))
        if len(lidar_files) == 0:
            errors.append(f"雷达源目录中没有文件: {lidar_pre_dir}")
    
    if errors:
        print("检查失败:")
        for error in errors:
            print(f"  错误: {error}")
        return False
    
    return True

def main():
    """
    主函数
    """
    print("=" * 60)
    print("图像和雷达文件重命名脚本")
    print("=" * 60)
    
    # 验证目录
    if not verify_directories():
        print("\n请检查源目录和文件是否存在！")
        return
    
    # 重命名并复制图像文件
    print("\n处理图像文件:")
    num_images = rename_and_copy_images()
    
    # 重命名并复制雷达文件
    print("\n处理雷达文件:")
    num_lidar = rename_and_copy_lidar()
    
    # 检查文件数量是否匹配
    if num_images != num_lidar:
        print(f"\n警告: 图像文件数量({num_images})与雷达文件数量({num_lidar})不匹配！")
        print("将使用较少的数量生成names.txt文件。")
    
    # 使用较少的数量
    num_files = min(num_images, num_lidar)
    
    # 创建names.txt文件
    create_names_file(num_files)
    
    print("\n" + "=" * 60)
    print("所有操作完成！")
    print(f"处理了 {num_files} 对文件")
    print("=" * 60)

if __name__ == "__main__":
    main()
