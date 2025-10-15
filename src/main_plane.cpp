#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include <message_filters/subscriber.h>
#include <geometry_msgs/PointStamped.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

// PCL相关头文件
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>

#include "DealString.h"
#include "FileIO.h"
#include "PCLlib.h"

using namespace std;

vector<pcl::PointXYZ> board_points;

// 点击回调函数
void waypointCallback(const geometry_msgs::PointStampedConstPtr& waypoint)
{
    pcl::PointXYZ board_point;
    board_point.x = waypoint->point.x;
    board_point.y = waypoint->point.y;
    board_point.z = waypoint->point.z;
    board_points.push_back(board_point);
    cout << "Clicked point: " << board_point << endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "plane_processor");
    ros::NodeHandle n;
    
    // 发布点云到rviz
    ros::Publisher map_pub = n.advertise<sensor_msgs::PointCloud2>("/MapCloud", 1);
    
    // 订阅点击的点
    ros::Subscriber waypoint_sub = n.subscribe("/clicked_point", 100, waypointCallback);
    
    // 设置路径
    string path_root;
    n.getParam("/plane_processor/root", path_root);
    string path_names = path_root + "/names.txt";
    string lidar_config_path;
    n.getParam("/plane_processor/lidar_config_path", lidar_config_path);
    
    // 输出目录
    string output_dir = path_root + "/lidar_plane";

    
    // 读取文件列表
    vector<string> imgnames = FileIO::ReadTxt2String(path_names, false);
    
    if (imgnames.empty())
    {
        cerr << "Error: No files found in " << path_names << endl;
        return -1;
    }
    
    cout << "Found " << imgnames.size() << " files to process" << endl;
    
    pcl::PointCloud<pcl::PointXYZI> global_map, target, source, choose_points;
    Eigen::Matrix4f Twl, Tlw;
    bool FirstLidarFrame = true;
    
    // NDT初始化
    PCLlib::NDT ndt(lidar_config_path);
    
    int success_count = 0;
    int fail_count = 0;
    
    for (int i = 0; i < imgnames.size(); i++)
    {
        target.clear();
        
        vector<string> name = read_format(imgnames[i], " ");
        
        // 读取点云文件
        string filename = path_root + "/lidar/" + name[0] + ".ply";
        cout << "\n[" << (i + 1) << "/" << imgnames.size() << "] Loading: " << filename << endl;
        
        target = Load_ply(filename);
        
        if (target.empty())
        {
            cerr << "Error: Failed to load or empty point cloud" << endl;
            fail_count++;
            continue;
        }
        
        cout << "Loaded " << target.size() << " points" << endl;
        
        // 等待rviz启动
        while (map_pub.getNumSubscribers() <= 0 && ros::ok())
        {
            cout << "Waiting for rviz to start..." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // 发布点云到rviz
        if (ros::ok())
        {
            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(target, map_msg);
            map_msg.header.frame_id = "/velodyne";
            map_msg.header.stamp = ros::Time::now();
            map_pub.publish(map_msg);
            cout << "Published point cloud to rviz" << endl;
        }
        
        pcl::PointXYZ seed_point;
        
        // 第一帧：等待用户点击种子点
        if (FirstLidarFrame)
        {
            board_points.clear();
            cout << "\n=== First Frame: Please click a seed point on the plane in rviz ===" << endl;
            cout << "Waiting for click..." << endl;
            
            while (board_points.empty() && ros::ok())
            {
                ros::spinOnce();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            if (!ros::ok() || board_points.empty())
            {
                cout << "No point clicked, exiting" << endl;
                break;
            }
            
            seed_point = board_points[0];
            cout << "Seed point received: (" << seed_point.x << ", " 
                 << seed_point.y << ", " << seed_point.z << ")" << endl;
        }
        else
        {
            // 后续帧：使用NDT变换种子点
            cout << "\nStarting NDT transformation for frame " << i << endl;
            ndt.AddCurrentCloud(target, Twl);
            Tlw = Twl.inverse();
            
            // 变换种子点到当前帧
            pcl::PointCloud<pcl::PointXYZI>::Ptr trans_ptr(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(choose_points, *trans_ptr, Tlw);
            
            if (!trans_ptr->points.empty())
            {
                seed_point.x = trans_ptr->points[0].x;
                seed_point.y = trans_ptr->points[0].y;
                seed_point.z = trans_ptr->points[0].z;
                cout << "Transformed seed point: (" << seed_point.x << ", " 
                     << seed_point.y << ", " << seed_point.z << ")" << endl;
            }
            else
            {
                cout << "Warning: NDT transformation failed, skipping this frame" << endl;
                fail_count++;
                continue;
            }
        }
        
        // 第一次RANSAC检测
        Eigen::Vector4f first_plane_model;
        Eigen::Vector3f center;
        
        cout << "\nStarting first RANSAC plane detection..." << endl;
        bool first_success = PCLlib::FirstRansacDetection(
            first_plane_model, center, target, seed_point,
            0.2, 0.05, 20);
        
        if (!first_success)
        {
            cout << "First RANSAC failed, skipping this file" << endl;
            fail_count++;
            continue;
        }
        
        // 发布第一次RANSAC结果
        if (ros::ok())
        {
            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(target, map_msg);
            map_msg.header.frame_id = "/velodyne";
            map_msg.header.stamp = ros::Time::now();
            map_pub.publish(map_msg);
            cout << "Published first RANSAC result (intensity 50)" << endl;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // 第二次RANSAC检测
        Eigen::Vector4f final_plane_model;
        cout << "\nStarting second RANSAC plane detection..." << endl;
        PCLlib::SecondRansacDetection(
            final_plane_model, target, seed_point, first_plane_model,
            0.4, 0.015, 0.05, 20, 50.0);  // 平面点强度设为50
        
        // 将非平面点的强度设为70
        cout << "Setting non-plane points intensity to 70..." << endl;
        int plane_count = 0;
        int other_count = 0;
        for (int j = 0; j < target.size(); j++)
        {
            if (target.points[j].intensity == 50.0)
            {
                plane_count++;
            }
            else
            {
                target.points[j].intensity = 70.0;
                other_count++;
            }
        }
        
        cout << "Updated intensities: " << plane_count << " plane points (50), " 
             << other_count << " other points (70)" << endl;
        
        // 发布第二次RANSAC结果
        if (ros::ok())
        {
            sensor_msgs::PointCloud2 map_msg2;
            pcl::toROSMsg(target, map_msg2);
            map_msg2.header.frame_id = "/velodyne";
            map_msg2.header.stamp = ros::Time::now();
            map_pub.publish(map_msg2);
            cout << "Published second RANSAC result" << endl;
        }
        
        // 保存处理后的点云
        string output_file = output_dir + "/" + name[0] + ".ply";
        cout << "\nSaving to: " << output_file << endl;
        
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
        if (pcl::io::savePLYFileBinary(output_file, target) == 0)
        {
            cout << "Successfully saved!" << endl;
            success_count++;
        }
        else
        {
            cerr << "Error: Failed to save file" << endl;
            fail_count++;
            continue;
        }
        
        // 第一帧处理完成后，初始化NDT
        if (FirstLidarFrame)
        {
            // 保存种子点用于后续NDT变换
            pcl::PointXYZI p;
            p.x = seed_point.x;
            p.y = seed_point.y;
            p.z = seed_point.z;
            p.intensity = 1.0;
            choose_points.push_back(p);
            
            source = target;
            ndt.AddCurrentCloud(source, Twl);
            FirstLidarFrame = false;
            cout << "\nFirst frame processed. NDT initialized for subsequent frames." << endl;
        }
        
        cout << "Frame " << i << " processing completed." << endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    

    return 0;
}
