# 雷达图像同步处理脚本

## 功能介绍

这个脚本实现了两个主要功能：

1. **PCD转TXT**: 将雷达PCD格式文件转换为TXT格式，包含x, y, z, intensity四个字段
2. **时间戳同步**: 根据雷达数据的时间戳找到最接近的图像，实现雷达-图像数据对齐

## 文件说明

- `lidar_image_sync.py` - 主要处理脚本
- `run_sync_example.py` - 交互式使用示例
- `README_sync.md` - 本说明文件


### 方法2：命令行参数

```bash
cd /home/lz/Camera-Lidar-Calibration-master/python
python3 lidar_image_sync.py --lidar_dir ../app/pointcloud_data --image_dir ../app/images --output_dir ./output
```

参数说明：
- `--lidar_dir`: 雷达PCD文件目录（必需）
- `--image_dir`: 图像文件目录（必需）
- `--output_dir`: 输出目录（可选，默认为output）

## 输入要求

### 文件命名格式
文件名必须包含时间戳信息，格式为：`年月日_时分秒_纳秒.扩展名`

示例：
- 雷达文件: `20250829_182646_136693000.pcd`
- 图像文件: `20250829_182646_170049190.png`

### 支持的文件格式
- **雷达文件**: `.pcd` (支持ASCII和Binary格式)
- **图像文件**: `.png`, `.jpg`, `.jpeg`

## 输出结果

脚本会在指定的输出目录下创建两个文件夹：

1. **lidar_txt/**: 包含转换后的TXT格式雷达文件
   - 格式：`x y z intensity`（空格分隔）
   - 文件名与原PCD文件对应

2. **aligned_images/**: 包含时间戳对齐后的图像文件
   - 文件名与对应的雷达文件名一致
   - 基于雷达时间戳选择最接近的图像

## 处理流程

1. 扫描雷达和图像文件夹，获取所有文件
2. 解析每个文件的时间戳信息
3. 对于每个雷达文件：
   - 读取PCD数据并转换为TXT格式
   - 根据时间戳找到最接近的图像文件
   - 复制对应图像到输出目录
4. 输出处理统计信息

## 注意事项

1. **时间戳格式**: 文件名必须严格按照`YYYYMMDD_HHMMSS_NNNNNNNNN`格式命名
2. **数据频率**: 脚本以雷达数据为基准，适用于雷达频率低于图像频率的情况
3. **PCD格式**: 支持包含x,y,z坐标和intensity信息的PCD文件
4. **内存使用**: 对于大型点云文件，请确保有足够的内存

## 示例用法

假设您有以下目录结构：
```
/home/lz/Camera-Lidar-Calibration-master/
├── app/
│   ├── pointcloud_data/  # 雷达PCD文件
│   └── images/          # 图像文件
└── python/
    ├── lidar_image_sync.py
    └── run_sync_example.py
```

运行示例脚本：
```bash
cd python
python3 run_sync_example.py
```

输入路径：
- 雷达PCD文件夹路径: `../app/pointcloud_data`
- 图像文件夹路径: `../app/images`
- 输出文件夹路径: `./output`

处理完成后，会在`python/output/`目录下生成：
- `lidar_txt/` - TXT格式的雷达文件
- `aligned_images/` - 对齐后的图像文件

## 错误处理

脚本会处理以下常见错误：
- 无效的时间戳格式
- 损坏的PCD文件
- 缺少必要字段的PCD文件
- 文件读取权限问题

如果遇到问题，请检查：
1. 文件路径是否正确
2. 文件命名是否符合时间戳格式
3. PCD文件是否包含x,y,z坐标信息
4. 是否有足够的磁盘空间
