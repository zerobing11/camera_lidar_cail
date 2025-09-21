#!/bin/bash

# 直接执行这个脚本就是启动容器，如果加上参数kill就是把启动的容器给删除掉
# Docker 镜像名称和标签
IMAGE_NAME="ros"
IMAGE_TAG="kinetic-robot"
CONTAINER_NAME='cali'

xhost +local:root

if [ $# -eq 0 ]; then
    # 检查Docker是否正在运行
    if ! docker info >/dev/null 2>&1; then
        echo "Docker 未运行，请启动 Docker 服务。"
        exit 1
    fi

    # 启动容器（如果容器已存在，则直接进入容器）
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
            echo "容器已存在且正在运行，直接进入..."
            docker exec -it $CONTAINER_NAME bash -c "source /opt/ros/kinetic/setup.bash && cd /home && exec bash"
        else
            echo "容器存在但已停止，启动容器..."
            docker start $CONTAINER_NAME
            docker exec -it $CONTAINER_NAME bash -c "source /opt/ros/kinetic/setup.bash && cd /home && exec bash"
        fi
    else
        echo "正在创建并启动容器 $CONTAINER_NAME..."
        
        # 创建并启动容器
        docker run -itd \
            --name "$CONTAINER_NAME" \
            --gpus all \
            --device /dev/tty\
            --device /dev/dri \
            --device /dev/nvidia0 \
            --device /dev/nvidiactl \
            --device /dev/nvidia-uvm \
            --group-add video \
            --network host \
            --privileged \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
            -e NVIDIA_VISIBLE_DEVICES=all \
            -e __NV_PRIME_RENDER_OFFLOAD=1 \
            -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
            -e DISPLAY=$DISPLAY \
            -e QT_X11_NO_MITSHM=1 \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "$(pwd)/":/home \
            --ipc=host \
            "${IMAGE_NAME}:${IMAGE_TAG}" \
            bash -c " bash"

        # 检查容器是否成功创建
        if [ $? -eq 0 ]; then
            echo "容器创建成功，正在进入..."
            docker exec -it $CONTAINER_NAME bash -c "source /opt/ros/kinetic/setup.bash && cd /home && exec bash"
        else
            echo "容器创建失败，请检查错误信息"
            exit 1
        fi
    fi
    
elif [ "$1" = "kill" ]; then
    echo "尝试停止并移除容器 '$CONTAINER_NAME'..."

    # 检查容器是否存在
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        # 停止容器
        docker stop $CONTAINER_NAME >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "容器已停止。"
        else
            echo "停止容器失败。"
            exit 1
        fi

        # 删除容器
        docker rm $CONTAINER_NAME >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "容器已移除。"
        else
            echo "移除容器失败。"
            exit 1
        fi
        
    else
        echo "容器不存在。"
        exit 1
    fi
    
elif [ "$1" = "commit" ]; then
    # 检查是否提供了新镜像名称
    if [ -z "$2" ]; then
        echo "错误：请指定新镜像名称（格式：新镜像名:标签）"
        echo "示例：$0 commit my_new_image:latest \"自定义提交信息\""
        exit 1
    fi
    
    NEW_IMAGE="$2"
    COMMIT_MESSAGE="${3:-从容器 $CONTAINER_NAME 提交}"  # 默认消息或自定义消息
    
    echo "正在提交容器 '$CONTAINER_NAME' 为镜像 '$NEW_IMAGE'..."
    echo "提交信息: $COMMIT_MESSAGE"
    
    # 检查容器是否存在
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        # 如果容器正在运行，停止它以保持一致性
        if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
            echo "容器正在运行，先停止容器以确保状态一致..."
            docker stop $CONTAINER_NAME >/dev/null
        fi
        
        # 提交容器为新镜像（带自定义消息）
        docker commit \
            --message "$COMMIT_MESSAGE" \
            $CONTAINER_NAME $NEW_IMAGE
        
        if [ $? -eq 0 ]; then
            echo "提交成功！新镜像信息："
            docker images | grep $(echo $NEW_IMAGE | cut -d':' -f1)
        else
            echo "提交失败，请检查错误信息"
            exit 1
        fi
    else
        echo "错误：容器 '$CONTAINER_NAME' 不存在"
        exit 1
    fi

else
    echo "无效参数。可用选项:"
    echo "无参数      - 启动/进入容器"
    echo "kill       - 停止并移除容器"
    echo "commit <镜像名:标签> - 提交容器为新镜像"
    exit 1
fi
