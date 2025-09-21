#!/usr/bin/env bash
set -e
# 载入 ROS Kinetic 环境
ibus-daemon -xdr
source /opt/ros/kinetic/setup.bash

# Ceres/CMake 路径（你的系统已确认）
export Ceres_DIR=/usr/lib/cmake/ceres
export CMAKE_PREFIX_PATH=/usr/lib/cmake:/opt/ros/kinetic:${CMAKE_PREFIX_PATH}
export CMAKE_MODULE_PATH=/usr/lib/cmake:${CMAKE_MODULE_PATH}

exec /opt/clion/bin/clion.sh
