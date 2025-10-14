#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达点云重命名脚本
将python/output/lidar_txt路径下的TXT点云文件重命名为lidar_001.txt格式，并复制到src/data/lidar目录下
"""

import os
import shutil
import glob
from pathlib import Path

def rename_and_copy_lidar():
    """
    重命名并复制雷达点云文件
    """
    # 源目录和目标目录
    source_dir = "/home/lz/camera_lidar_cail/src/data/lidar_pre"
    target_dir = "/home/lz/camera_lidar_cail/src/data/lidar"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有TXT文件
    lidar_files = glob.glob(os.path.join(source_dir, "*.txt"))
    
    # 按文件名排序，确保顺序一致
    lidar_files.sort()
    
    print(f"找到 {len(lidar_files)} 个TXT点云文件")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print("-" * 50)
    
    # 重命名并复制文件
    for i, lidar_file in enumerate(lidar_files, 1):
        # 生成新的文件名
        new_filename = f"lidar_{i:03d}.txt"
        target_path = os.path.join(target_dir, new_filename)
        
        # 复制文件
        try:
            shutil.copy2(lidar_file, target_path)
            print(f"复制: {os.path.basename(lidar_file)} -> {new_filename}")
        except Exception as e:
            print(f"错误: 复制 {lidar_file} 失败 - {e}")
    
    print("-" * 50)
    print(f"完成！已处理 {len(lidar_files)} 个文件")
    print(f"文件已保存到: {target_dir}")

def create_lidar_names_file():
    """
    创建lidar_names.txt文件，包含雷达文件的映射关系
    """
    names_file = "/home/lz/camera_lidar_cail/src/data/lidar_names.txt"
    
    # 获取雷达文件数量
    target_dir = "/home/lz/camera_lidar_cail/src/data/lidar"
    lidar_files = glob.glob(os.path.join(target_dir, "lidar_*.txt"))
    num_lidar = len(lidar_files)
    
    print(f"\n创建lidar_names.txt文件，包含 {num_lidar} 个雷达文件映射...")
    
    with open(names_file, 'w', encoding='utf-8') as f:
        f.write("# 格式：雷达文件名 编号\n")
        f.write("# 每行格式：lidar_001 001\n")
        f.write("# 其中lidar_001是雷达文件名（不包含.txt扩展名），001是编号\n\n")
        
        for i in range(1, num_lidar + 1):
            lidar_name = f"lidar_{i:03d}"
            lidar_num = f"{i:03d}"
            f.write(f"{lidar_name} {lidar_num}\n")
    
    print(f"lidar_names.txt文件已创建: {names_file}")

def verify_file_order():
    """
    验证文件顺序是否正确
    """
    source_dir = "/home/lz/camera_lidar_cail/python/output/lidar_txt"
    target_dir = "/home/lz/camera_lidar_cail/src/data/lidar"
    
    # 获取源文件和目标文件列表
    source_files = sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "lidar_*.txt")))
    
    print(f"\n验证文件顺序...")
    print(f"源文件数量: {len(source_files)}")
    print(f"目标文件数量: {len(target_files)}")
    
    if len(source_files) != len(target_files):
        print("警告: 源文件和目标文件数量不匹配！")
        return False
    
    # 验证前几个文件的对应关系
    print("\n前5个文件的对应关系:")
    for i in range(min(5, len(source_files))):
        source_name = os.path.basename(source_files[i])
        target_name = os.path.basename(target_files[i])
        expected_name = f"lidar_{i+1:03d}.txt"
        
        if target_name == expected_name:
            print(f"✓ {source_name} -> {target_name}")
        else:
            print(f"✗ {source_name} -> {target_name} (期望: {expected_name})")
            return False
    
    print("✓ 文件顺序验证通过！")
    return True

def main():
    """
    主函数
    """
    print("=" * 60)
    print("雷达点云重命名和复制脚本")
    print("=" * 60)
    
    # 检查源目录是否存在
    source_dir = "/home/lz/camera_lidar_cail/python/output/lidar_txt"
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在 - {source_dir}")
        print("请先运行 lidar_image_sync.py 生成TXT格式的雷达文件")
        return
    
    # 检查是否有TXT文件
    txt_files = glob.glob(os.path.join(source_dir, "*.txt"))
    if len(txt_files) == 0:
        print(f"错误: 源目录中没有找到TXT文件 - {source_dir}")
        return
    
    # 重命名并复制雷达文件
    rename_and_copy_lidar()
    
    # 创建雷达文件映射
    create_lidar_names_file()
    
    # 验证文件顺序
    verify_file_order()
    
    print("\n" + "=" * 60)
    print("所有操作完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
