#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS时间戳雷达-图像同步

功能：
- 以雷达频率为基准，按文件名中的ROS时间戳匹配最接近的图像
- 保留原始PLY与图像文件名，将匹配结果分别拷贝到输出目录下

要求的时间戳文件名格式示例：
- 图像/雷达：1758870611.349549770.png / 1758870611.349549770.ply
使用示例
python3 /home/lz/camera_lidar_cail/python/lidar_img_sync.py \
  --lidar_dir /home/lz/camera_lidar_cail/xxxxx \
  --image_dir /home/lz/camera_lidar_cail/xxxxx \
  --output_dir /home/lz/camera_lidar_cail/src/data_/output
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


def collect_images_with_timestamps(image_dir: str) -> List[Tuple[str, int]]:
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


def find_closest_image(lidar_ts_ns: int, images_with_ts: List[Tuple[str, int]]) -> Tuple[str, int]:
    min_diff = float('inf')
    closest = None
    for img_path, img_ts in images_with_ts:
        diff = abs(lidar_ts_ns - img_ts)
        if diff < min_diff:
            min_diff = diff
            closest = img_path
    return closest, int(min_diff if min_diff != float('inf') else -1)


def sync_lidar_image_with_ply(lidar_dir: str, image_dir: str, output_dir: str) -> None:
    print(f"雷达目录: {lidar_dir}")
    print(f"图像目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    out_lidar_dir = os.path.join(output_dir, "lidar_pre")
    out_image_dir = os.path.join(output_dir, "img_pre")
    os.makedirs(out_lidar_dir, exist_ok=True)
    os.makedirs(out_image_dir, exist_ok=True)

    lidar_files = glob.glob(os.path.join(lidar_dir, "*.ply"))
    lidar_files.sort()
    if not lidar_files:
        print("错误：未找到PLY雷达文件！")
        return

    images_with_ts = collect_images_with_timestamps(image_dir)
    print(f"找到 {len(lidar_files)} 个雷达文件")
    print(f"成功解析 {len(images_with_ts)} 个带有效时间戳的图像文件")
    if not images_with_ts:
        print("错误：未解析到任何有效图像时间戳，终止")
        return

    matched_count = 0
    for idx, ply_path in enumerate(lidar_files, start=1):
        try:
            lidar_ts_ns = parse_ros_timestamp_ns_from_filename(ply_path)
        except Exception as e:
            print(f"跳过 {os.path.basename(ply_path)} ：无法解析时间戳（{e}）")
            continue

        closest_image, diff_ns = find_closest_image(lidar_ts_ns, images_with_ts)
        print(f"处理 {idx}/{len(lidar_files)}: {os.path.basename(ply_path)}")
        if closest_image is None:
            print("  警告：未找到匹配图像")
            continue

        # 拷贝PLY
        ply_name = os.path.basename(ply_path)
        out_ply = os.path.join(out_lidar_dir, ply_name)
        shutil.copy2(ply_path, out_ply)

        # 拷贝图像（保留原名）
        img_name = os.path.basename(closest_image)
        out_img = os.path.join(out_image_dir, img_name)
        shutil.copy2(closest_image, out_img)

        print(f"  对齐图像: {img_name} (Δt={diff_ns/1e6:.2f} ms)")
        matched_count += 1

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"成功同步 {matched_count} 对雷达-图像")
    print(f"输出PLY目录: {out_lidar_dir}")
    print(f"输出图像目录: {out_image_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description='ROS时间戳雷达-图像同步（使用PLY，不生成TXT）')
    parser.add_argument('--lidar_dir', required=True, type=str, help='雷达PLY目录')
    parser.add_argument('--image_dir', required=True, type=str, help='图像目录（文件名需含ROS时间戳）')
    parser.add_argument('--output_dir', default='output_ros_ply', type=str, help='输出根目录')

    args = parser.parse_args()

    if not os.path.isdir(args.lidar_dir):
        print(f"错误：雷达目录不存在 - {args.lidar_dir}")
        return
    if not os.path.isdir(args.image_dir):
        print(f"错误：图像目录不存在 - {args.image_dir}")
        return

    sync_lidar_image_with_ply(args.lidar_dir, args.image_dir, args.output_dir)


if __name__ == '__main__':
    main()


