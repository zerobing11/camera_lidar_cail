#!/usr/bin/env bash
set -e
# 载入 ROS Kinetic 环境
#ibus-daemon -xdr
source /opt/ros/kinetic/setup.bash

# Ceres/CMake 路径（你的系统已确认）
export Ceres_DIR=/usr/lib/cmake/ceres
export CMAKE_PREFIX_PATH=/usr/lib/cmake:/opt/ros/kinetic:${CMAKE_PREFIX_PATH}
export CMAKE_MODULE_PATH=/usr/lib/cmake:${CMAKE_MODULE_PATH}

# 确保CLion能找到ROS头文件
export ROS_ROOT=/opt/ros/kinetic/share/ros
export ROS_PACKAGE_PATH=/opt/ros/kinetic/share
export PYTHONPATH=/opt/ros/kinetic/lib/python2.7/dist-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=/opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=/opt/ros/kinetic/lib/pkgconfig:/opt/ros/kinetic/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH}

# 打印环境变量用于调试
echo "ROS_ROOT: $ROS_ROOT"
echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
echo "catkin_INCLUDE_DIRS should include: /opt/ros/kinetic/include"

exec /opt/clion/bin/clion.sh
