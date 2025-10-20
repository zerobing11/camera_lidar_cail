cmake_minimum_required( VERSION 2.8 )
project( camera_calibration )

set( CMAKE_BUILD_TYPE "Debug" )
# 添加c++ 11标准支持
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g -O2")
set(OpenCV_DIR "/usr/local/OpenCV345/share/OpenCV")

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs cv_bridge sensor_msgs pcl_ros message_generation)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)

catkin_package()

include_directories(
		    ${PROJECT_SOURCE_DIR}
		    ${PROJECT_SOURCE_DIR}/src/ThirdParty/
		    ${OpenCV_INCLUDE_DIRS}
		    ${catkin_INCLUDE_DIRS} 
		    "/usr/include/eigen3"
		    ${CERES_INCLUDE_DIRS}
		    ${PCL_INCLUDE_DIRS}
		    )

add_executable( camera_calibration src/main.cpp)
add_executable( camera_calibration_muti src/main_muti.cpp)

# 链接所有的库
target_link_libraries( camera_calibration 
		      ${OpenCV_LIBS}
			  ${CERES_LIBRARIES}
		      ${catkin_LIBRARIES}
			  ${PCL_LIBRARIES}
		      ${PROJECT_SOURCE_DIR}/src/ThirdParty/build/libKITTICornerDetection.so)
target_link_libraries( camera_calibration_muti
		${OpenCV_LIBS}
		${CERES_LIBRARIES}
		${catkin_LIBRARIES}
		${PCL_LIBRARIES}
		${PROJECT_SOURCE_DIR}/src/ThirdParty/build/libKITTICornerDetection.so)
## 添加平面处理节点
add_executable( plane_processor src/main_plane.cpp)
target_link_libraries( plane_processor
		      ${OpenCV_LIBS}
		      ${catkin_LIBRARIES}
			  ${PCL_LIBRARIES})