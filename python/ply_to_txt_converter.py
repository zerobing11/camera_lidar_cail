#!/usr/bin/env python3
"""
PLY to TXT Converter
将PLY文件转换为TXT格式，格式与lidar_old文件夹中的文件一致


python3 ply_to_txt_converter.py --help
获取使用方法



"""

import os
import struct
import argparse
from pathlib import Path


def read_ply_file(ply_path):
    """
    读取PLY文件并返回点云数据
    
    Args:
        ply_path (str): PLY文件路径
        
    Returns:
        list: 包含(x, y, z, intensity)元组的列表
    """
    points = []
    
    with open(ply_path, 'rb') as f:
        # 读取头部信息
        header_lines = []
        line = f.readline().decode('ascii').strip()
        header_lines.append(line)
        
        if line != 'ply':
            raise ValueError("不是有效的PLY文件")
        
        vertex_count = 0
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line == 'end_header':
                break
        
        # 读取二进制数据
        for i in range(vertex_count):
            # 读取4个float值: x, y, z, intensity
            data = f.read(16)  # 4 * 4 bytes = 16 bytes
            if len(data) < 16:
                break
                
            x, y, z, intensity = struct.unpack('<ffff', data)
            points.append((x, y, z, intensity))
    
    return points


def write_txt_file(points, txt_path):
    """
    将点云数据写入TXT文件
    
    Args:
        points (list): 包含(x, y, z, intensity)元组的列表
        txt_path (str): 输出TXT文件路径
    """
    with open(txt_path, 'w') as f:
        # 写入头部
        f.write("x y z intensity\n")
        
        # 写入点云数据
        for x, y, z, intensity in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {intensity:.6f}\n")


def convert_ply_to_txt(ply_path, output_dir=None):
    """
    将单个PLY文件转换为TXT文件
    
    Args:
        ply_path (str): PLY文件路径
        output_dir (str): 输出目录，如果为None则使用PLY文件所在目录
    """
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY文件不存在: {ply_path}")
    
    if output_dir is None:
        output_dir = ply_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    txt_filename = ply_path.stem + '.txt'
    txt_path = output_dir / txt_filename
    
    print(f"转换: {ply_path.name} -> {txt_path.name}")
    
    try:
        # 读取PLY文件
        points = read_ply_file(ply_path)
        print(f"  读取到 {len(points)} 个点")
        
        # 写入TXT文件
        write_txt_file(points, txt_path)
        print(f"  成功写入: {txt_path}")
        
    except Exception as e:
        print(f"  错误: {e}")


def batch_convert_ply_to_txt(input_dir, output_dir=None):
    """
    批量转换目录中的所有PLY文件
    
    Args:
        input_dir (str): 包含PLY文件的输入目录
        output_dir (str): 输出目录，如果为None则使用输入目录
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有PLY文件
    ply_files = list(input_dir.glob("*.ply"))
    
    if not ply_files:
        print(f"在目录 {input_dir} 中没有找到PLY文件")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    success_count = 0
    for ply_file in sorted(ply_files):
        try:
            convert_ply_to_txt(ply_file, output_dir)
            success_count += 1
        except Exception as e:
            print(f"转换失败 {ply_file.name}: {e}")
    
    print("-" * 50)
    print(f"转换完成: {success_count}/{len(ply_files)} 个文件成功")


def main():
    parser = argparse.ArgumentParser(description="将PLY文件转换为TXT格式")
    parser.add_argument("input", help="输入PLY文件或包含PLY文件的目录")
    parser.add_argument("-o", "--output", help="输出目录（可选）")
    parser.add_argument("--single", action="store_true", help="转换单个文件而不是批量转换")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.single or input_path.is_file():
        # 转换单个文件
        if not input_path.suffix.lower() == '.ply':
            print("错误: 输入文件必须是PLY格式")
            return
        
        convert_ply_to_txt(input_path, args.output)
    else:
        # 批量转换
        batch_convert_ply_to_txt(input_path, args.output)


if __name__ == "__main__":
    main()
