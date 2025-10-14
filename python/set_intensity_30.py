#!/usr/bin/env python3
# coding: utf-8

import os
import sys
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


def process_file(path: str) -> int:
    """将 ASCII PLY 文件的 vertex 强度字段设置为 30。
    仅处理 format ascii 的 .ply；若为 binary 则跳过。
    返回被修改的顶点数量。
    """
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
        # 非 ASCII 或无法解析，跳过
        return 0

    # 查找 intensity 属性索引
    try:
        intensity_idx = vertex_props.index("intensity")
    except ValueError:
        # 无 intensity 字段，跳过
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
        if os.path.isabs(arg_dir):
            target_dir = arg_dir
        else:
            target_dir = os.path.abspath(os.path.join(os.getcwd(), arg_dir))

    if not os.path.isdir(target_dir):
        print("目录不存在:", target_dir)
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
