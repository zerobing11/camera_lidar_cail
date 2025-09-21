#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
names.txt填充脚本
根据src/data目录下的雷达和图像数据，按照names.txt的格式要求填充映射关系
"""

import os
import glob
from pathlib import Path

def scan_data_files():
    """
    扫描src/data目录下的雷达和图像文件
    """
    # 定义路径
    data_dir = "/home/lz/Camera-Lidar-Calibration-master/src/data"
    lidar_dir = os.path.join(data_dir, "lidar")
    image_dir = os.path.join(data_dir, "leftImg")
    names_file = os.path.join(data_dir, "names.txt")
    
    print(f"数据目录: {data_dir}")
    print(f"雷达目录: {lidar_dir}")
    print(f"图像目录: {image_dir}")
    print(f"映射文件: {names_file}")
    print("=" * 60)
    
    # 检查目录是否存在
    if not os.path.exists(lidar_dir):
        print(f"错误: 雷达目录不存在 - {lidar_dir}")
        return None, None, None
    
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在 - {image_dir}")
        return None, None, None
    
    # 扫描雷达文件
    lidar_files = glob.glob(os.path.join(lidar_dir, "lidar_*.txt"))
    lidar_files.sort()
    
    # 扫描图像文件
    image_files = glob.glob(os.path.join(image_dir, "left_*.png"))
    image_files.sort()
    
    print(f"找到 {len(lidar_files)} 个雷达文件")
    print(f"找到 {len(image_files)} 个图像文件")
    
    return lidar_files, image_files, names_file

def extract_file_number(filename):
    """
    从文件名中提取编号
    例如: lidar_001.txt -> 001, left_001.png -> 001
    """
    basename = os.path.basename(filename)
    # 提取下划线后的数字部分
    parts = basename.split('_')
    if len(parts) >= 2:
        number_part = parts[1].split('.')[0]  # 去掉扩展名
        return number_part
    return None

def fill_names_file(lidar_files, image_files, names_file):
    """
    填充names.txt文件
    """
    print("\n开始填充names.txt文件...")
    
    # 创建雷达文件编号到文件名的映射
    lidar_map = {}
    for lidar_file in lidar_files:
        number = extract_file_number(lidar_file)
        if number:
            lidar_name = f"lidar_{number}"
            lidar_map[number] = lidar_name
    
    # 创建图像文件编号到文件名的映射
    image_map = {}
    for image_file in image_files:
        number = extract_file_number(image_file)
        if number:
            image_map[number] = number  # 图像编号就是数字部分
    
    print(f"雷达文件映射: {len(lidar_map)} 个")
    print(f"图像文件映射: {len(image_map)} 个")
    
    # 找到共同的编号
    common_numbers = set(lidar_map.keys()) & set(image_map.keys())
    common_numbers = sorted(common_numbers)
    
    print(f"共同编号: {len(common_numbers)} 个")
    
    if len(common_numbers) == 0:
        print("错误: 没有找到匹配的雷达和图像文件编号")
        return False
    
    # 写入names.txt文件
    try:
        with open(names_file, 'w', encoding='utf-8') as f:
            # 写入头部注释
            f.write("# 格式：lidar文件名 图像编号\n")
            f.write("# 每行格式：lidar_xxx 001\n")
            f.write("# 其中lidar_xxx是lidar文件名（不包含.txt扩展名），001是图像编号(不含left)\n\n")
            
            # 写入映射关系
            for number in common_numbers:
                lidar_name = lidar_map[number]
                image_number = image_map[number]
                f.write(f"{lidar_name} {image_number}\n")
        
        print(f"✓ names.txt文件已成功填充")
        print(f"✓ 共写入 {len(common_numbers)} 条映射关系")
        return True
        
    except Exception as e:
        print(f"错误: 写入names.txt文件失败 - {e}")
        return False

def verify_mapping(lidar_files, image_files, names_file):
    """
    验证映射关系的正确性
    """
    print("\n验证映射关系...")
    
    # 读取names.txt文件
    try:
        with open(names_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 过滤掉注释行和空行
        mapping_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                mapping_lines.append(line)
        
        print(f"names.txt中包含 {len(mapping_lines)} 条映射关系")
        
        # 验证前几条映射
        print("\n前5条映射关系:")
        for i, line in enumerate(mapping_lines[:5]):
            parts = line.split()
            if len(parts) == 2:
                lidar_name, image_number = parts
                print(f"  {i+1}. {lidar_name} -> {image_number}")
            else:
                print(f"  {i+1}. 格式错误: {line}")
        
        # 检查文件是否存在
        print("\n检查文件存在性...")
        missing_files = []
        
        for line in mapping_lines:
            parts = line.split()
            if len(parts) == 2:
                lidar_name, image_number = parts
                
                # 检查雷达文件
                lidar_file = os.path.join("/home/lz/Camera-Lidar-Calibration-master/src/data/lidar", f"{lidar_name}.txt")
                if not os.path.exists(lidar_file):
                    missing_files.append(f"雷达文件: {lidar_file}")
                
                # 检查图像文件
                image_file = os.path.join("/home/lz/Camera-Lidar-Calibration-master/src/data/leftImg", f"left_{image_number}.png")
                if not os.path.exists(image_file):
                    missing_files.append(f"图像文件: {image_file}")
        
        if missing_files:
            print("警告: 发现缺失的文件:")
            for missing in missing_files[:10]:  # 只显示前10个
                print(f"  - {missing}")
            if len(missing_files) > 10:
                print(f"  ... 还有 {len(missing_files) - 10} 个文件缺失")
        else:
            print("✓ 所有映射的文件都存在")
        
        return True
        
    except Exception as e:
        print(f"错误: 验证映射关系失败 - {e}")
        return False

def show_statistics(lidar_files, image_files):
    """
    显示统计信息
    """
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    
    # 雷达文件统计
    print(f"雷达文件总数: {len(lidar_files)}")
    if lidar_files:
        print(f"第一个雷达文件: {os.path.basename(lidar_files[0])}")
        print(f"最后一个雷达文件: {os.path.basename(lidar_files[-1])}")
    
    # 图像文件统计
    print(f"图像文件总数: {len(image_files)}")
    if image_files:
        print(f"第一个图像文件: {os.path.basename(image_files[0])}")
        print(f"最后一个图像文件: {os.path.basename(image_files[-1])}")
    
    # 编号范围
    if lidar_files and image_files:
        lidar_numbers = [extract_file_number(f) for f in lidar_files]
        image_numbers = [extract_file_number(f) for f in image_files]
        
        lidar_numbers = [n for n in lidar_numbers if n]
        image_numbers = [n for n in image_numbers if n]
        
        if lidar_numbers:
            print(f"雷达文件编号范围: {min(lidar_numbers)} - {max(lidar_numbers)}")
        if image_numbers:
            print(f"图像文件编号范围: {min(image_numbers)} - {max(image_numbers)}")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("names.txt填充脚本")
    print("=" * 60)
    
    # 扫描数据文件
    lidar_files, image_files, names_file = scan_data_files()
    
    if lidar_files is None or image_files is None:
        return
    
    # 显示统计信息
    show_statistics(lidar_files, image_files)
    
    # 填充names.txt文件
    success = fill_names_file(lidar_files, image_files, names_file)
    
    if success:
        # 验证映射关系
        verify_mapping(lidar_files, image_files, names_file)
        
        print("\n" + "=" * 60)
        print("✓ names.txt文件填充完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ names.txt文件填充失败！")
        print("=" * 60)

if __name__ == "__main__":
    main()
