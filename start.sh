source /opt/ros/kinetic/setup.bash
source build/devel/setup.bash
rm -rf src/data/img_cornor_test
roslaunch camera_calibration start.launch
