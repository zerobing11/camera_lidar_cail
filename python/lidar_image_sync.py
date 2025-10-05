#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达图像同步脚本
功能：
1. 将PCD格式雷达文件转换为TXT格式（包含x, y, z, intensity信息）
2. 基于时间戳同步雷达和图像数据，以雷达频率为准
"""

import os
import glob
import shutil
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import struct

def parse_timestamp(filename):
    """
    从文件名中解析时间戳
    格式：20250829_182646_136693000.xxx
    返回：datetime对象和纳秒部分
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # 分割时间戳：年月日_时分秒_纳秒
    parts = basename.split('_')
    if len(parts) != 3:
        raise ValueError(f"无效的时间戳格式: {basename}")
    
    date_str = parts[0]  # 20250829
    time_str = parts[1]  # 182646
    nano_str = parts[2]  # 136693000
    
    # 解析日期和时间
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    hour = int(time_str[:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    
    # 创建datetime对象
    dt = datetime(year, month, day, hour, minute, second)
    nanoseconds = int(nano_str)
    
    return dt, nanoseconds

def timestamp_to_total_nanoseconds(dt, nanoseconds):
    """
    将datetime和纳秒转换为总纳秒数（用于比较）
    """
    # 转换为Unix时间戳（秒）
    timestamp_seconds = dt.timestamp()
    # 转换为纳秒
    total_nanoseconds = int(timestamp_seconds * 1e9) + nanoseconds
    return total_nanoseconds

def read_pcd_file(pcd_file):
    """
    读取PCD文件并返回点云数据 (x, y, z, intensity)
    支持ASCII和Binary格式的PCD文件
    """
    points = []
    
    with open(pcd_file, 'rb') as f:
        header_lines = []
        
        # 读取头部信息
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header_lines.append(line)
            
            if line.startswith('DATA'):
                data_format = line.split()[1]  # ascii 或 binary
                break
        
        # 解析头部信息
        width = None
        height = None
        points_count = None
        fields = []
        size = []
        type_info = []
        
        for line in header_lines:
            if line.startswith('WIDTH'):
                width = int(line.split()[1])
            elif line.startswith('HEIGHT'):
                height = int(line.split()[1])
            elif line.startswith('POINTS'):
                points_count = int(line.split()[1])
            elif line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                size = [int(x) for x in line.split()[1:]]
            elif line.startswith('TYPE'):
                type_info = line.split()[1:]
        
        # 确定字段索引
        x_idx = y_idx = z_idx = intensity_idx = None
        try:
            x_idx = fields.index('x')
            y_idx = fields.index('y')
            z_idx = fields.index('z')
            # 尝试找到intensity字段（可能是'intensity'或'i'）
            if 'intensity' in fields:
                intensity_idx = fields.index('intensity')
            elif 'i' in fields:
                intensity_idx = fields.index('i')
        except ValueError:
            raise ValueError(f"PCD文件缺少必要的坐标字段: {pcd_file}")
        
        # 读取数据
        if data_format.lower() == 'ascii':
            # ASCII格式
            for line in f:
                line = line.decode('utf-8', errors='ignore').strip()
                if line:
                    values = line.split()
                    if len(values) >= len(fields):
                        x = float(values[x_idx])
                        y = float(values[y_idx])
                        z = float(values[z_idx])
                        intensity = float(values[intensity_idx]) if intensity_idx is not None else 0.0
                        points.append([x, y, z, intensity])
        
        elif data_format.lower() == 'binary':
            # Binary格式
            point_size = sum(size)
            
            # 创建格式字符串
            format_chars = []
            for i, (s, t) in enumerate(zip(size, type_info)):
                if t == 'F':  # float
                    if s == 4:
                        format_chars.append('f')
                    elif s == 8:
                        format_chars.append('d')
                elif t == 'I':  # unsigned int
                    if s == 1:
                        format_chars.append('B')
                    elif s == 2:
                        format_chars.append('H')
                    elif s == 4:
                        format_chars.append('I')
                elif t == 'U':  # unsigned
                    if s == 1:
                        format_chars.append('B')
                    elif s == 2:
                        format_chars.append('H')
                    elif s == 4:
                        format_chars.append('I')
            
            format_str = '<' + ''.join(format_chars)  # little endian
            
            # 读取所有点
            for _ in range(points_count):
                data = f.read(point_size)
                if len(data) < point_size:
                    break
                
                try:
                    values = struct.unpack(format_str, data)
                    x = float(values[x_idx])
                    y = float(values[y_idx])
                    z = float(values[z_idx])
                    intensity = float(values[intensity_idx]) if intensity_idx is not None else 0.0
                    points.append([x, y, z, intensity])
                except struct.error:
                    continue
    
    return np.array(points)

def save_txt_file(points, txt_file):
    """
    保存点云数据为TXT格式 (x y z intensity)
    """
    np.savetxt(txt_file, points, fmt='%.6f', delimiter=' ', 
               header='x y z intensity', comments='')

def find_closest_image(lidar_timestamp_ns, image_files_with_timestamps):
    """
    找到与雷达时间戳最接近的图像文件
    """
    min_diff = float('inf')
    closest_image = None
    
    for image_file, image_timestamp_ns in image_files_with_timestamps:
        diff = abs(lidar_timestamp_ns - image_timestamp_ns)
        if diff < min_diff:
            min_diff = diff
            closest_image = image_file
    
    return closest_image, min_diff

def process_lidar_image_sync(lidar_dir, image_dir, output_dir):
    """
    处理雷达图像同步
    """
    print(f"雷达目录: {lidar_dir}")
    print(f"图像目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 创建输出目录
    lidar_txt_dir = os.path.join(output_dir, "lidar_txt")
    aligned_image_dir = os.path.join(output_dir, "aligned_images")
    
    os.makedirs(lidar_txt_dir, exist_ok=True)
    os.makedirs(aligned_image_dir, exist_ok=True)
    
    # 获取所有雷达文件
    lidar_files = glob.glob(os.path.join(lidar_dir, "*.pcd"))
    lidar_files.sort()
    
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))
    image_files.sort()
    
    print(f"找到 {len(lidar_files)} 个雷达文件")
    print(f"找到 {len(image_files)} 个图像文件")
    
    if len(lidar_files) == 0:
        print("错误：未找到雷达文件！")
        return
    
    if len(image_files) == 0:
        print("错误：未找到图像文件！")
        return
    
    # 解析图像文件时间戳
    print("\n解析图像文件时间戳...")
    image_files_with_timestamps = []
    
    for image_file in image_files:
        try:
            dt, nanoseconds = parse_timestamp(image_file)
            timestamp_ns = timestamp_to_total_nanoseconds(dt, nanoseconds)
            image_files_with_timestamps.append((image_file, timestamp_ns))
        except ValueError as e:
            print(f"警告：跳过无效时间戳的图像文件 {image_file}: {e}")
    
    print(f"成功解析 {len(image_files_with_timestamps)} 个图像文件的时间戳")
    
    # 处理每个雷达文件
    print("\n开始处理雷达文件...")
    processed_count = 0
    
    for i, lidar_file in enumerate(lidar_files):
        try:
            # 解析雷达文件时间戳
            dt, nanoseconds = parse_timestamp(lidar_file)
            lidar_timestamp_ns = timestamp_to_total_nanoseconds(dt, nanoseconds)
            
            # 读取PCD文件
            print(f"处理 {i+1}/{len(lidar_files)}: {os.path.basename(lidar_file)}")
            points = read_pcd_file(lidar_file)
            
            if len(points) == 0:
                print(f"  警告：雷达文件为空，跳过")
                continue
            
            # 保存为TXT格式
            lidar_basename = os.path.splitext(os.path.basename(lidar_file))[0]
            txt_filename = f"{lidar_basename}.txt"
            txt_filepath = os.path.join(lidar_txt_dir, txt_filename)
            save_txt_file(points, txt_filepath)
            print(f"  保存TXT: {txt_filename} ({len(points)} 个点)")
            
            # 找到最接近的图像
            closest_image, time_diff_ns = find_closest_image(lidar_timestamp_ns, image_files_with_timestamps)
            
            if closest_image:
                # 复制对应的图像
                image_basename = os.path.basename(closest_image)
                aligned_image_path = os.path.join(aligned_image_dir, f"{lidar_basename}.png")
                
                shutil.copy2(closest_image, aligned_image_path)
                time_diff_ms = time_diff_ns / 1e6  # 转换为毫秒
                print(f"  对齐图像: {image_basename} (时间差: {time_diff_ms:.2f}ms)")
                
                processed_count += 1
            else:
                print(f"  警告：未找到匹配的图像")
        
        except Exception as e:
            print(f"  错误：处理 {lidar_file} 失败 - {e}")
    
    print("\n" + "=" * 60)
    print(f"处理完成！")
    print(f"成功处理 {processed_count} 对雷达-图像数据")
    print(f"雷达TXT文件保存到: {lidar_txt_dir}")
    print(f"对齐图像保存到: {aligned_image_dir}")
    print("=" * 60)

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='雷达图像同步处理脚本')
    parser.add_argument('--lidar_dir', type=str, required=True,
                        help='雷达PCD文件目录路径')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='图像文件目录路径')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录路径 (默认: output)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.lidar_dir):
        print(f"错误：雷达目录不存在 - {args.lidar_dir}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"错误：图像目录不存在 - {args.image_dir}")
        return
    
    # 处理数据
    process_lidar_image_sync(args.lidar_dir, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()
