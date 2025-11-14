#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS时间戳相机-雷达同步（以相机频率为基准）

功能：
- 以相机频率为基准，按文件名中的ROS时间戳匹配最接近的雷达数据
- 保留原始PLY与图像文件名，将匹配结果分别拷贝到输出目录下

要求的时间戳文件名格式示例：
- 图像/雷达：1758870611.349549770.png / 1758870611.349549770.ply

使用示例：
python3 /home/lz/Camera-Lidar-Calibration-master/python/camera_lidar_sync.py \
  --lidar_dir /home/lz/Camera-Lidar-Calibration-master/data/lidar_pre \
  --image_dir /home/lz/Camera-Lidar-Calibration-master/data/img_pre \
  --output_dir /home/lz/Camera-Lidar-Calibration-master/data/output
"""

import os
import glob
import shutil
import argparse
from typing import List, Tuple


def parse_ros_timestamp_ns_from_filename(path: str) -> int:
    """
    从文件名解析 ROS 时间戳，格式：<seconds>.<nanoseconds>
    返回：总纳秒（int）
    例如："1758870611.349549770.png" -> 1758870611349549770
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    parts = basename.split('.')
    if len(parts) != 2:
        raise ValueError(f"无效的ROS时间戳格式: {basename}")
    seconds = int(parts[0])
    nanoseconds = int(parts[1])
    return seconds * 1_000_000_000 + nanoseconds


def collect_lidar_with_timestamps(lidar_dir: str) -> List[Tuple[str, int]]:
    """收集所有雷达文件及其时间戳"""
    lidar_files = []
    lidar_files.extend(glob.glob(os.path.join(lidar_dir, "*.ply")))
    lidar_files.sort()

    lidar_with_ts: List[Tuple[str, int]] = []
    for lidar in lidar_files:
        try:
            ts_ns = parse_ros_timestamp_ns_from_filename(lidar)
            lidar_with_ts.append((lidar, ts_ns))
        except Exception:
            # 跳过无法解析的文件名
            continue
    return lidar_with_ts


def collect_images_with_timestamps(image_dir: str) -> List[Tuple[str, int]]:
    """收集所有图像文件及其时间戳"""
    image_files = []
    image_files.extend(glob.glob(os.path.join(image_dir, "*.png")))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))
    image_files.sort()

    images_with_ts: List[Tuple[str, int]] = []
    for img in image_files:
        try:
            ts_ns = parse_ros_timestamp_ns_from_filename(img)
            images_with_ts.append((img, ts_ns))
        except Exception:
            # 跳过无法解析的文件名
            continue
    return images_with_ts


def find_closest_lidar(image_ts_ns: int, lidar_with_ts: List[Tuple[str, int]]) -> Tuple[str, int]:
    """为给定的图像时间戳找到最接近的雷达数据"""
    min_diff = float('inf')
    closest = None
    for lidar_path, lidar_ts in lidar_with_ts:
        diff = abs(image_ts_ns - lidar_ts)
        if diff < min_diff:
            min_diff = diff
            closest = lidar_path
    return closest, int(min_diff if min_diff != float('inf') else -1)


def sync_camera_lidar_with_ply(lidar_dir: str, image_dir: str, output_dir: str) -> None:
    """以相机频率为基准进行雷达-图像同步"""
    print(f"雷达目录: {lidar_dir}")
    print(f"图像目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 创建输出目录
    out_lidar_dir = os.path.join(output_dir, "lidar")
    out_image_dir = os.path.join(output_dir, "img")
    os.makedirs(out_lidar_dir, exist_ok=True)
    os.makedirs(out_image_dir, exist_ok=True)

    # 收集雷达和图像文件
    lidar_with_ts = collect_lidar_with_timestamps(lidar_dir)
    images_with_ts = collect_images_with_timestamps(image_dir)
    
    print(f"找到 {len(lidar_with_ts)} 个雷达文件")
    print(f"找到 {len(images_with_ts)} 个图像文件")
    
    if not lidar_with_ts:
        print("错误：未找到PLY雷达文件！")
        return
    if not images_with_ts:
        print("错误：未找到图像文件！")
        return

    matched_count = 0
    total_time_diff = 0
    max_time_diff = 0
    min_time_diff = float('inf')
    
    print("\n开始同步处理...")
    print("-" * 60)
    
    for idx, (img_path, img_ts_ns) in enumerate(images_with_ts, start=1):
        try:
            closest_lidar, diff_ns = find_closest_lidar(img_ts_ns, lidar_with_ts)
            print(f"处理 {idx}/{len(images_with_ts)}: {os.path.basename(img_path)}")
            
            if closest_lidar is None:
                print("  警告：未找到匹配的雷达数据")
                continue

            # 拷贝图像（保留原名）
            img_name = os.path.basename(img_path)
            out_img = os.path.join(out_image_dir, img_name)
            shutil.copy2(img_path, out_img)

            # 拷贝雷达数据（保留原名）
            lidar_name = os.path.basename(closest_lidar)
            out_lidar = os.path.join(out_lidar_dir, lidar_name)
            shutil.copy2(closest_lidar, out_lidar)

            diff_ms = diff_ns / 1e6
            print(f"  匹配雷达: {lidar_name} (Δt={diff_ms:.2f} ms)")
            
            matched_count += 1
            total_time_diff += diff_ms
            max_time_diff = max(max_time_diff, diff_ms)
            min_time_diff = min(min_time_diff, diff_ms)
            
        except Exception as e:
            print(f"  错误：处理 {os.path.basename(img_path)} 时出错: {e}")
            continue

    print("\n" + "=" * 60)
    print("同步处理完成！")
    print(f"成功同步 {matched_count} 对相机-雷达数据")
    print(f"输出雷达目录: {out_lidar_dir}")
    print(f"输出图像目录: {out_image_dir}")
    
    if matched_count > 0:
        avg_time_diff = total_time_diff / matched_count
        print(f"\n时间同步统计:")
        print(f"  平均时间差: {avg_time_diff:.2f} ms")
        print(f"  最大时间差: {max_time_diff:.2f} ms")
        print(f"  最小时间差: {min_time_diff:.2f} ms")
    
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description='ROS时间戳相机-雷达同步（以相机频率为基准）')
    parser.add_argument('--lidar_dir', required=True, type=str, help='雷达PLY目录')
    parser.add_argument('--image_dir', required=True, type=str, help='图像目录（文件名需含ROS时间戳）')
    parser.add_argument('--output_dir', default='output_camera_sync', type=str, help='输出根目录')

    args = parser.parse_args()

    if not os.path.isdir(args.lidar_dir):
        print(f"错误：雷达目录不存在 - {args.lidar_dir}")
        return
    if not os.path.isdir(args.image_dir):
        print(f"错误：图像目录不存在 - {args.image_dir}")
        return

    sync_camera_lidar_with_ply(args.lidar_dir, args.image_dir, args.output_dir)


if __name__ == '__main__':
    main()
