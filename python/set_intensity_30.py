#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import struct
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 使用相对路径，脚本位于 python 目录下，目标为 ../data/lidar
LIDAR_DIR = os.path.join(ROOT, "data", "lidar")


def list_ply_files(directory: str) -> List[str]:
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.lower().endswith(".ply"):
            files.append(path)
    files.sort()
    return files


def process_binary_ply(path: str) -> int:
    """处理二进制PLY文件，将强度字段设置为30.0"""
    with open(path, "rb") as f:
        # 读取头部
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == "end_header":
                break
        
        # 解析头部信息
        format_binary = False
        vertex_count = None
        vertex_props = []
        in_vertex_props = False
        
        for line in header_lines:
            line_lower = line.lower()
            if line_lower.startswith("format "):
                if "binary" in line_lower:
                    format_binary = True
            elif line_lower.startswith("element vertex"):
                try:
                    vertex_count = int(line.split()[2])
                except Exception:
                    vertex_count = None
                in_vertex_props = True
                vertex_props = []
            elif line_lower.startswith("element ") and not line_lower.startswith("element vertex"):
                in_vertex_props = False
            elif line_lower.startswith("property") and in_vertex_props:
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append(parts[2])
        
        if not format_binary or vertex_count is None:
            return 0
        
        # 查找intensity属性索引
        try:
            intensity_idx = vertex_props.index("intensity")
        except ValueError:
            return 0
        
        # 计算每个顶点的字节大小
        vertex_size = len(vertex_props) * 4  # 假设所有属性都是float (4字节)
        
        # 读取所有顶点数据
        vertex_data = bytearray(f.read(vertex_count * vertex_size))
        if len(vertex_data) != vertex_count * vertex_size:
            return 0
        
        # 修改强度值
        changed = 0
        for i in range(vertex_count):
            offset = i * vertex_size + intensity_idx * 4
            # 将强度值设置为30.0
            struct.pack_into('<f', vertex_data, offset, 30.0)
            changed += 1
        
        # 写回文件
        if changed > 0:
            with open(path, "rb+") as f:
                # 重新计算头部长度
                header_bytes = 0
                for line in header_lines:
                    header_bytes += len(line.encode('ascii')) + 1  # +1 for newline
                f.seek(header_bytes)  # 跳过头部
                f.write(vertex_data)
        
        return changed


def detect_ply_format(path: str) -> str:
    """检测PLY文件格式：'ascii' 或 'binary'"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_lower = line.strip().lower()
                if line_lower.startswith("format "):
                    if "ascii" in line_lower:
                        return "ascii"
                    elif "binary" in line_lower:
                        return "binary"
                elif line_lower == "end_header":
                    break
    except Exception:
        pass
    return "unknown"


def process_file(path: str) -> int:
    """处理PLY文件，将强度字段设置为30.0
    支持ASCII和二进制格式
    返回被修改的顶点数量。
    """
    # 检测文件格式
    format_type = detect_ply_format(path)
    
    if format_type == "binary":
        return process_binary_ply(path)
    elif format_type == "ascii":
        return process_ascii_ply(path)
    else:
        print(f"警告: 无法识别PLY格式: {os.path.basename(path)}")
        return 0


def process_ascii_ply(path: str) -> int:
    """处理ASCII格式的PLY文件"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines:
        return 0

    # 解析 PLY 头
    if not lines[0].strip().lower().startswith("ply"):
        return 0

    format_ascii = False
    vertex_count = None
    vertex_props: List[str] = []
    in_vertex_props = False
    header_end_idx = None

    i = 1
    while i < len(lines):
        line = lines[i].strip().lower()
        if line.startswith("format "):
            # e.g. format ascii 1.0
            if "ascii" in line:
                format_ascii = True
        elif line.startswith("element vertex"):
            # e.g. element vertex 1000
            try:
                vertex_count = int(line.split()[2])
            except Exception:
                vertex_count = None
            in_vertex_props = True
            vertex_props = []
        elif line.startswith("element ") and not line.startswith("element vertex"):
            in_vertex_props = False
        elif line.startswith("property") and in_vertex_props:
            # e.g. property float x
            parts = line.split()
            if len(parts) >= 3:
                vertex_props.append(parts[2])
        elif line == "end_header":
            header_end_idx = i
            break
        i += 1

    if header_end_idx is None or not format_ascii or vertex_count is None:
        return 0

    # 查找 intensity 属性索引
    try:
        intensity_idx = vertex_props.index("intensity")
    except ValueError:
        return 0

    # 修改顶点强度
    changed = 0
    start = header_end_idx + 1
    end = min(start + vertex_count, len(lines))
    for j in range(start, end):
        s = lines[j].rstrip("\n")
        if not s:
            continue
        parts = s.split()
        if len(parts) <= intensity_idx:
            continue
        parts[intensity_idx] = "30.000000"
        lines[j] = " ".join(parts) + "\n"
        changed += 1

    # 写回文件
    if changed > 0:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    return changed


def main():
    # 允许用户传自定义目录；否则使用默认目录
    target_dir = LIDAR_DIR
    
    if len(sys.argv) > 1:
        arg_dir = sys.argv[1]
        # 直接使用 os.path.abspath() 处理路径，无论输入是绝对路径还是相对路径
        target_dir = os.path.abspath(arg_dir)
        print(f"使用指定目录: {target_dir}")
    else:
        print(f"使用默认目录: {target_dir}")

    if not os.path.isdir(target_dir):
        print("错误: 目录不存在:", target_dir)
        print("用法: python set_intensity_30.py [目录路径]")
        print("示例: python set_intensity_30.py /home/user/data/lidar")
        print("示例: python set_intensity_30.py ../data/lidar")
        sys.exit(1)

    ply_files = list_ply_files(target_dir)
    if not ply_files:
        print("未找到PLY文件:", target_dir)
        sys.exit(0)

    total_changed = 0
    for fp in ply_files:
        c = process_file(fp)
        print(f"处理 {os.path.basename(fp)}: 修改 {c} 个顶点")
        total_changed += c

    print(f"完成。共修改 {total_changed} 个顶点，文件数 {len(ply_files)}，目录: {target_dir}")


if __name__ == "__main__":
    main()
