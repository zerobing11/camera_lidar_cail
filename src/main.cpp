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
#include "sensor_msgs/Image.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/PointField.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> //为了求特征值

// 先包含PCL相关头文件，避免与OpenCV的flann命名空间冲突
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/search/search.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/surface/convex_hull.h>

#include "DealString.h"
#include "FileIO.h"
#include "ICP.h"
#include "PCLlib.h"
#include "Camera.h"
#include "Segment.h"

// 最后包含OpenCV头文件
#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

#define corner_height 3
#define corner_width 4

cv::Mat distParameter;//摄像机的畸变系数
cv::Mat intrincMatrix;
map<int,vector<Eigen::Vector4f>> lidar_planes;
map<int,vector<Eigen::Vector4f>> cam_planes;//根据world plane和相机的位姿矩阵得到相机坐标系下的平面坐标，顺序是右左下
vector<vector<Eigen::Vector3f>> lidarCenters;
vector<int> camera_valid_frame;
int max_thread;

string path_root;
string lidar_config_path;
string path_names;
std::mutex mutexpcd;
bool use_NDT=true;
bool first_frame=true;
bool showPCD=true;
bool FirstLidarFrame=true;
bool wait_enter=false;

// RANSAC参数
double first_ransac_radius = 0.3;
double first_ransac_threshold = 0.05;
double second_ransac_radius = 0.7;
double second_ransac_y_distance_threshold = 0.02;
double second_ransac_threshold = 0.015;

vector<pcl::PointXYZ> board_points;

void waypointCallback(const geometry_msgs::PointStampedConstPtr& waypoint)
{
  //成功接收到了点的坐标信息
  pcl::PointXYZ board_point;
  board_point.x=waypoint->point.x;
  board_point.y=waypoint->point.y;
  board_point.z=waypoint->point.z;
  board_points.push_back(board_point);
  cout<<board_point<<endl;
}

void calculateExtrinsicError(string planes_path,string extrinsic_path)
{
    vector<vector<float>> content = FileIO::ReadTxt2Float(extrinsic_path,true);//path_Ext
    Eigen::Matrix3f Rcl_float;
    Eigen::Vector3f tcl_float;
    cv::Mat K_cv = Mat(3,3,CV_64FC1,Scalar::all(0));//摄像机的畸变系数
    cv::Mat distParameter_res = Mat(1,5,CV_64FC1,Scalar::all(0));//摄像机的畸变系数
    Rcl_float(0,0) = content[0][0];Rcl_float(0,1) = content[0][1];Rcl_float(0,2) = content[0][2];
    Rcl_float(1,0) = content[1][0];Rcl_float(1,1) = content[1][1];Rcl_float(1,2) = content[1][2];
    Rcl_float(2,0) = content[2][0];Rcl_float(2,1) = content[2][1];Rcl_float(2,2) = content[2][2];
    tcl_float[0] = content[3][0];
    tcl_float[1] = content[3][1];
    tcl_float[2] = content[3][2];

    vector<vector<float>> cal_planes = FileIO::ReadTxt2Float(planes_path);
    vector<Eigen::Vector4f> temp_pc,temp_pl;
    vector<vector<Eigen::Vector4f>> lidarPlanes;
	vector<vector<Eigen::Vector4f>> camPlanes;
    for(int j=0;j<cal_planes.size()/3;j++)
    {
        for(int i=0;i<3;i++)
        {
            float a = cal_planes[3*j+i][0];
            float b = cal_planes[3*j+i][1];
            float c = cal_planes[3*j+i][2];
            float d = cal_planes[3*j+i][3];
            temp_pl.push_back(Eigen::Vector4f(a,b,c,d));
            a = cal_planes[3*j+i][4];
            b = cal_planes[3*j+i][5];
            c = cal_planes[3*j+i][6];
            d = cal_planes[3*j+i][7];
            temp_pc.push_back(Eigen::Vector4f(a,b,c,d));
        }
        lidarPlanes.push_back(temp_pl);
        camPlanes.push_back(temp_pc);
        temp_pl.clear();
        temp_pc.clear();
    }
    int n=lidarPlanes.size();
    float n_error=0,d_error=0,maxN=0,maxD=0,ind1,ind2;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<3;j++)
        {
            Eigen::Vector3f nl(lidarPlanes[i][j][0],lidarPlanes[i][j][1],lidarPlanes[i][j][2]);
            Eigen::Vector3f nc(camPlanes[i][j][0],camPlanes[i][j][1],camPlanes[i][j][2]);
            Eigen::Vector3f nlc=Rcl_float*nl;
            float axb,a,b;
            axb=nlc[0]*nc[0]+nlc[1]*nc[1]+nlc[2]*nc[2];
            a=sqrt(nlc[0]*nlc[0]+nlc[1]*nlc[1]+nlc[2]*nlc[2]);
            b=sqrt(nc[0]*nc[0]+nc[1]*nc[1]+nc[2]*nc[2]);
            float error=acos(axb/(a*b));
            n_error+=error;

            float cam_D = lidarPlanes[i][j][3]+(nl[0]*tcl_float[0]+nl[1]*tcl_float[1]+nl[2]*tcl_float[2]);
            float error2=camPlanes[i][j][3]-cam_D;
            d_error+=abs(error2);
            cout<<i<<" "<<j<<" "<<error/PI*180<<" "<<error2<<endl;
            if(error>maxN)
            {
                ind1=i;
                ind2=j;
                maxN=error;
            }
            if(error2>maxD)
            {
                maxD=error2;
            }
        }
        cout<<endl;
    }
    cout<<"mean error normal="<<(n_error/3/n)/PI*180<<endl;
    cout<<"mean error distance="<<d_error/3/n<<endl;
    cout<<"index:"<<ind1<<"boards:"<<ind2<<"max error normal="<<maxN/PI*180<<endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "camera_calibration");
	ros::NodeHandle n;
    ros::Publisher map_pub = n.advertise<sensor_msgs::PointCloud2>("/MapCloud",1);//聚类的点云
    nav_msgs::Path path_msg;
    ros::Publisher path_pub = n.advertise<nav_msgs::Path>("/Path", 1);//轨迹
    ros::Publisher pose_pub = n.advertise<geometry_msgs::PoseStamped>("/Pose", 1);//ndt输出得到的pose
    //设置路径
	n.getParam("/camera_calibration/root", path_root);//根目录
    string path_coor = path_root+"/3D.txt";
	string path_coor_planar = path_root+"/3D_planar.txt";
	path_names = path_root+"/names.txt";
    //设置程序标定模式
    int getIntrincMatrix,getExtrinsic;
    n.getParam("/camera_calibration/IntrincMatrix", getIntrincMatrix);//是否标定相机
	n.getParam("/camera_calibration/Extrinsic", getExtrinsic);//
    n.getParam("/camera_calibration/lidar_config_path", lidar_config_path);
    // 设置平移求解方式（true: 使用平面中心点约束; false: 使用平面距离约束）
    bool use_plane_center_translation = true;
    n.getParam("/camera_calibration/use_plane_center_translation", use_plane_center_translation);

    //设置角点检测和棋盘格识别的阈值参数
    double corner_detect_threshold = 0.15;  // 角点检测阈值默认值
    double chessboard_threshold = 0.8;      // 棋盘格识别阈值默认值
    n.getParam("/camera_calibration/corner_detect_threshold", corner_detect_threshold);
    n.getParam("/camera_calibration/chessboard_threshold", chessboard_threshold);
    
    //设置是否启用按回车键等待功能
    n.getParam("/camera_calibration/wait_enter", wait_enter);
    
    //设置RANSAC参数
    n.getParam("/camera_calibration/first_ransac_radius", first_ransac_radius);
    n.getParam("/camera_calibration/first_ransac_threshold", first_ransac_threshold);
    n.getParam("/camera_calibration/second_ransac_radius", second_ransac_radius);
    n.getParam("/camera_calibration/second_ransac_y_distance_threshold", second_ransac_y_distance_threshold);
    n.getParam("/camera_calibration/second_ransac_threshold", second_ransac_threshold);


    //标定
    if(getIntrincMatrix)
    {
        CAMERA::Camera Cam(corner_height,corner_width,path_root,corner_detect_threshold,chessboard_threshold);
        vector<string> lidar_img_names = FileIO::ReadTxt2String(path_names,false);
        for(int i=0;i<lidar_img_names.size();i++)//载入标定图像imgnames.size()
        {
            vector<string> lidar_img_name = read_format(lidar_img_names[i]," ");
            string img_name=path_root+"/img/"+lidar_img_name[1]+".png";
            bool ischoose=Cam.add(img_name);
            if(ischoose)
            {
                camera_valid_frame.push_back(i);
                cout<<"有效帧 " << camera_valid_frame.size() << ": " << lidar_img_name[1] << ".png" << endl;
            }
        }
        // cv::Mat predefined_intrinsic = (cv::Mat_<double>(3,3) <<
        //     606.3696899414062, 0.0, 417.6044616699219,
        //     0.0, 604.7097778320312, 247.67758178710938,
        //     0.0, 0.0, 1.0);
        //
        // cv::Mat predefined_distortion = (cv::Mat_<double>(1,5) <<
        //     0.0, 0.0, 0.0, 0.0, 0.0);
        cv::Mat predefined_intrinsic = (cv::Mat_<double>(3,3) <<
                909.5545654296875, 0.0, 630.40673828125,
                0.0, 907.0646362304688, 371.5163879394531,
                0.0, 0.0, 1.0);

        cv::Mat predefined_distortion = (cv::Mat_<double>(1,5) <<
            0.0, 0.0, 0.0, 0.0, 0.0);

        Cam.calibration(predefined_intrinsic, predefined_distortion);//使用预定义内参并计算外参
        Cam.show();//显示标定结果
        Cam.GetIntrincMatrix(intrincMatrix);
        Cam.GetDistParameter(distParameter);
        std::cout<<"intrinsic:"<<intrincMatrix<<std::endl;
        std::cout<<"distortion:"<<distParameter<<std::endl;
        Cam.GetPlanesModels(cam_planes);//获取标定板平面模型
        cv::destroyAllWindows();
    }

    //lidar
    if(getExtrinsic)
    {
        ros::Subscriber waypoint_sub_ = n.subscribe("/clicked_point", 100, waypointCallback);
        pcl::PointCloud<pcl::PointXYZI> global_map,target,source,choose_points;
        Eigen::Matrix4f Twl,Tlw;
        Eigen::Vector4f plane_model;
        Eigen::Vector3f center;
        PCLlib::NDT ndt(lidar_config_path);//ndt初始化构造函数
        vector<string> lidar_img_names = FileIO::ReadTxt2String(path_names,false);
        vector<Eigen::Vector4f> tmp_plane_params;
        vector<Eigen::Vector3f> plane_center;
        for(int i=0;i<camera_valid_frame.size();i++)
        {
            target.clear();

            int ind=camera_valid_frame[i];
            vector<string> img_lidar_name = read_format(lidar_img_names[ind]," ");//读取点云文件
            // 直接读取PLY点云文件
            string lidar_name=path_root+"/lidar/"+img_lidar_name[0]+".ply";
            target=Load_ply(lidar_name);
            std::cout<<"target.size():"<<target.size()<<std::endl;

            cout<<"------------Processing frame: "<<ind<<"------------"<<endl;
            if(!FirstLidarFrame)
            {
                ndt.AddCurrentCloud(target,Twl);
                Tlw=Twl.inverse();
                pcl::PointCloud<pcl::PointXYZI>::Ptr trans_ptr(new pcl::PointCloud<pcl::PointXYZI>());//当前帧变换到世界地图中的坐标
			    pcl::transformPointCloud(choose_points, *trans_ptr, Tlw);//将当前帧的点云scan_ptr变化到世界地图下得到transformed_scan_ptr

                for(int j=0;j<3;j++)
                {
                    pcl::PointXYZ Bpoint;
                    Bpoint.x=trans_ptr->points[j].x;
                    Bpoint.y=trans_ptr->points[j].y;
                    Bpoint.z=trans_ptr->points[j].z;
                    cout<<"Processing plane "<<(j+1)<<" at point ("<<Bpoint.x<<", "<<Bpoint.y<<", "<<Bpoint.z<<")"<<endl;

                    Eigen::Vector4f first_plane_model;
                    Eigen::Vector3f plane_center_single;
                    
                    // 第一次RANSAC检测，并计算平面中心点
                    bool first_success = PCLlib::FirstRansacDetection(first_plane_model, plane_center_single, 
                                                                     target, Bpoint,
                                                                     first_ransac_radius, first_ransac_threshold, 20);
                    
                    // 发布第一次RANSAC结果到rviz
                    if(first_success && ros::ok())
                    {
                        sensor_msgs::PointCloud2 map_msg;
                        pcl::toROSMsg(target, map_msg);
                        map_msg.header.frame_id = "/velodyne";
                        map_msg.header.stamp = ros::Time::now();
                        map_pub.publish(map_msg);
                        cout<<"Published first RANSAC result for plane "<<(j+1)<<" (intensity 50)"<<endl;
                        
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        
                        // 第二次RANSAC检测
                        float current_plane_intensity = 80.0 + (j * 20.0);
                        PCLlib::SecondRansacDetection(plane_model, target, Bpoint,first_plane_model,
                            second_ransac_radius, second_ransac_y_distance_threshold, second_ransac_threshold, 20, current_plane_intensity);

                        tmp_plane_params.push_back(plane_model);
                        plane_center.push_back(plane_center_single);
                        
                        // 发布第二次RANSAC结果到rviz
                        if(ros::ok())
                        {
                            sensor_msgs::PointCloud2 map_msg2;
                            pcl::toROSMsg(target, map_msg2);
                            map_msg2.header.frame_id = "/velodyne";
                            map_msg2.header.stamp = ros::Time::now();
                            map_pub.publish(map_msg2);
                            cout<<"Published second RANSAC result for plane "<<(j+1)<<" (intensity "<<current_plane_intensity<<")"<<endl;
                        }
                        
                        // 每个平面处理完后等待一段时间
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }
                    else
                    {
                        cout<<"Failed to detect plane for point "<<j<<" in frame "<<ind<<endl;
                        
                        // 即使失败也发布当前状态
                        if(ros::ok())
                        {
                            sensor_msgs::PointCloud2 map_msg;
                            pcl::toROSMsg(target, map_msg);
                            map_msg.header.frame_id = "/velodyne";
                            map_msg.header.stamp = ros::Time::now();
                            map_pub.publish(map_msg);
                        }
                    }
                }
                
                // 检查是否检测到有效的标定板平面
                if(PCLlib::CheckBoardPlane(tmp_plane_params))
                {
                    cout<<i<<" index:"<<ind<<" is choosen"<<endl<<endl;
                    lidar_planes.insert(pair<int,vector<Eigen::Vector4f>>(ind,tmp_plane_params));
                    lidarCenters.push_back(plane_center);
                    cout<<"The numbers of lidar-camera-calibration now:"<<lidar_planes.size()<<endl;
                    
                    // 发布最终结果
                    if(ros::ok())
                    {
                        sensor_msgs::PointCloud2 final_msg;
                        pcl::toROSMsg(target, final_msg);
                        final_msg.header.frame_id = "/velodyne";
                        final_msg.header.stamp = ros::Time::now();
                        map_pub.publish(final_msg);
                        cout<<"Published final result for frame "<<ind<<endl;
                    }
                }
                else
                {
                    cout<<"Frame "<<ind<<" rejected - planes do not form valid chessboard pattern"<<endl;
                }
                
                tmp_plane_params.clear();
                plane_center.clear();
                
                if(wait_enter)
                {
                    cout<<"Frame "<<ind<<" processing completed. Press ENTER to continue to next frame..."<<endl;
                    cin.get();
                }
                else
                {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            while(FirstLidarFrame)
            {
                while(map_pub.getNumSubscribers()<=0){//等待rviz完全启动
                std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                if(ros::ok())//全局地图
                {
                    sensor_msgs::PointCloud2 map_msg;
                    pcl::toROSMsg(target,map_msg);
                    map_msg.header.frame_id = "/velodyne";
                    map_msg.header.stamp = ros::Time::now();
                    map_pub.publish(map_msg);
                }
                ros::spinOnce();
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if(board_points.size()>0&&board_points.size()!=tmp_plane_params.size())
                {
                    float current_plane_intensity = 80.0 + (tmp_plane_params.size() * 20.0);

                    Eigen::Vector4f first_plane_model;
                    
                    // 第一次RANSAC检测，并计算平面中心点
                    bool first_success = PCLlib::FirstRansacDetection(first_plane_model, center, 
                                                                     target, board_points[tmp_plane_params.size()],
                                                                     first_ransac_radius, first_ransac_threshold, 20);
                    
                    // 发布第一次RANSAC结果
                    if(first_success && ros::ok())
                    {
                        sensor_msgs::PointCloud2 map_msg;
                        pcl::toROSMsg(target, map_msg);
                        map_msg.header.frame_id = "/velodyne";
                        map_msg.header.stamp = ros::Time::now();
                        map_pub.publish(map_msg);
 
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        
                        // 第二次RANSAC
                        PCLlib::SecondRansacDetection(plane_model, target, board_points[tmp_plane_params.size()],
                                                     first_plane_model, second_ransac_radius, second_ransac_y_distance_threshold, second_ransac_threshold, 20, current_plane_intensity);
                        
                        plane_center.push_back(center);
                        tmp_plane_params.push_back(plane_model);
                        
                        // 发布第二次RANSAC结果
                        if(ros::ok())
                        {
                            sensor_msgs::PointCloud2 map_msg2;
                            pcl::toROSMsg(target, map_msg2);
                            map_msg2.header.frame_id = "/velodyne";
                            map_msg2.header.stamp = ros::Time::now();
                            map_pub.publish(map_msg2);
                        }
                    }
                    else
                    {
                        cout<<"First RANSAC failed for plane "<<(tmp_plane_params.size()+1)<<endl;
                        
                        // 即使失败也发布当前状态
                        if(ros::ok())
                        {
                            sensor_msgs::PointCloud2 map_msg;
                            pcl::toROSMsg(target, map_msg);
                            map_msg.header.frame_id = "/velodyne";
                            map_msg.header.stamp = ros::Time::now();
                            map_pub.publish(map_msg);
                        }
                    }
                }
                if(tmp_plane_params.size()==3)
                {
                    bool choose=PCLlib::CheckBoardPlane(tmp_plane_params);
                    if(choose)
                    {
                        cout<<i<<" index:"<<ind<<" is choosen"<<endl<<endl;
                        lidar_planes.insert(pair<int,vector<Eigen::Vector4f>>(ind,tmp_plane_params));
                        lidarCenters.push_back(plane_center);
                        cout<<"The numbers of lidar-camera-calibration now:"<<lidar_planes.size()<<endl;
                    }
                    for(int i=0;i<3;i++)
                    {
                        pcl::PointXYZI p;
                        p.x=board_points[i].x;
                        p.y=board_points[i].y;
                        p.z=board_points[i].z;
                        choose_points.push_back(p);
                    }
                    
                    // 根据参数决定是否等待用户按回车键继续到下一帧
                    if(wait_enter)
                    {
                        cout<<"Press ENTER to continue to next frame..."<<endl;
                        cin.get(); // 等待用户按回车键
                    }
                    else
                    {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }
                    
                    FirstLidarFrame=false;
                    source=target;
                    ndt.AddCurrentCloud(source,Twl);
                    tmp_plane_params.clear();
                    plane_center.clear();
                    cout<<"choose board points finish"<<endl;
                    break;
                }
            }
        }
        cout<<"get valid lidar planes:"<<lidar_planes.size()<<endl;

        Eigen::Matrix3d Rcl_;
	    Eigen::Vector3d Tcl_;
        if(getExtrinsic)
        {
            cout<<"0.Start Extrinsic Optimization!!!!!!!!!!!"<<endl;
            vector<vector<Eigen::Vector4f>> lidarPlanes;
            vector<vector<Eigen::Vector4f>> camPlanes;
            // 与参与外参优化的平面一一对应的雷达系平面中心点
            vector<vector<Eigen::Vector3f>> lidarCentersOpt;

            int center_idx = 0;
            for(map<int,vector<Eigen::Vector4f>>::iterator iter=lidar_planes.begin();iter!=lidar_planes.end();iter++)
            {   
                int frame_idx = iter->first;
                // 这一frame同时检测到了相机和激光的平面，才会加入到优化中
                if((cam_planes.find(frame_idx)!=cam_planes.end())&&(lidar_planes.find(frame_idx)!=lidar_planes.end()))
                {
                    lidarPlanes.push_back(lidar_planes[frame_idx]);
                    camPlanes.push_back(cam_planes[frame_idx]);

                    // 使用与该frame对应的雷达平面中心点
                    if(center_idx < (int)lidarCenters.size())
                    {
                        lidarCentersOpt.push_back(lidarCenters[center_idx]);
                    }
                    else
                    {
                        cout<<"[Warning] lidarCenters index out of range when building lidarCentersOpt. frame_idx = "<<frame_idx<<endl;
                    }
                }
                // 无论是否加入优化，都向前推进一次索引，以保持与lidar_planes插入顺序一致
                center_idx++;
            }
            cout<<"Num of frames to optimize extrisinc = "<<lidarPlanes.size()<<endl;
            cout<<"lidarCenters.size():"<<lidarCenters.size()<<" "<<lidarCenters[0].size()<<endl;
            //为了计算得到tc,L和Rc,L
            //先拟合得到Rc,l 然后再线性计算得到tc,l
            if(lidarPlanes.size()!=0)
            {
                cout<<"1.Start Rotatoin Matrix Optimization!!!!!"<<endl;
                Eigen::Matrix3d Rcl;
                Eigen::Vector3d tcl,pl,plc;
                vector<Point3d> lidar_normals;
                vector<Point3d> cam_normals;
                Eigen::MatrixXd A;
                Eigen::MatrixXd b;
                A = Eigen::MatrixXd::Zero(3*lidarPlanes.size(),3);
                b = Eigen::MatrixXd::Zero(3*lidarPlanes.size(),1);
                for(int i=0;i<lidarPlanes.size();i++)
                {
                    for(int j=0;j<3;j++)
                    {
                        lidar_normals.push_back(cv::Point3d((double)lidarPlanes[i][j](0),(double)lidarPlanes[i][j](1),(double)lidarPlanes[i][j](2)));
                        cam_normals.push_back(cv::Point3d((double)camPlanes[i][j](0),(double)camPlanes[i][j](1),(double)camPlanes[i][j](2)));
                        A(3*i+j,0) = (double)camPlanes[i][j](0);
                        A(3*i+j,1) = (double)camPlanes[i][j](1);
                        A(3*i+j,2) = (double)camPlanes[i][j](2);
                    }
                }
                ICP::pose_estimation_3d3d (cam_normals,lidar_normals,Rcl,tcl);
                cout<<"ICP tcl(should be equal to zero) = "<<tcl.transpose()<<endl;
                cout<<"Final Rcl = "<<endl<<Rcl<<endl;
                cout<<"2.Start translation vector Optimization!!!!!"<<endl;
                if(use_plane_center_translation)
                {
                    cout<<"Using plane center constraint method"<<endl;
                    // 基于平面中心点的方法：n_c^T (R_cl * p_l + t_cl) + d_c = 0
                    // 推导得到：n_c^T t_cl = -( d_c + n_c^T R_cl p_l )
                    for(int i=0;i<lidarPlanes.size();i++)
                    {
                        for(int j=0;j<3;j++)
                        {
                            const Eigen::Vector3f& center = lidarCentersOpt[i][j];
                            Eigen::Vector3d pl(
                                (double)center(0),
                                (double)center(1),
                                (double)center(2));
                            Eigen::Vector3d plc = Rcl * pl;

                            double nx = (double)camPlanes[i][j](0);
                            double ny = (double)camPlanes[i][j](1);
                            double nz = (double)camPlanes[i][j](2);
                            double dc = (double)camPlanes[i][j](3);

                            b(3*i+j,0) = -( dc + nx*plc[0] + ny*plc[1] + nz*plc[2] );
                        }
                    }
                }
                else
                {
                    cout<<"Using direct plane distance constraint method"<<endl;
                    // 使用直接的平面距离约束：n_c^T * t_cl = d_c - d_l
                    for(int i=0;i<lidarPlanes.size();i++)
                    {
                        for(int j=0;j<3;j++)
                        {
                            b(3*i+j,0) = (double)camPlanes[i][j](3) - (double)lidarPlanes[i][j](3);
                        }
                    }
                }

                Eigen::EigenSolver<Eigen::MatrixXd> es( A.transpose()*A );
                cout<<"A = "<<endl<<A<<endl;
                cout<<"ATA = "<<endl<<A.transpose()*A<<endl;
                cout<<"b = "<<endl<<b<<endl;
                cout<<"ATA 矩阵的特征值 = "<<endl<<es.eigenvalues()<<endl;

                // 检查A矩阵的条件数
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
                cout<<"A matrix condition number = "<<cond<<endl;
                if(cond > 1e5) {
                    cout<<"Warning: A matrix is ill-conditioned! condition number = "<<cond<<endl;
                }
                Eigen::MatrixXd tcl_xd = A.colPivHouseholderQr().solve(b);
                
                // 检查求解结果的有效性
                if(!tcl_xd.allFinite()) {
                    cout<<"Error: Translation solution contains NaN or Inf values!"<<endl;
                    cout<<"tcl_xd = "<<tcl_xd.transpose()<<endl;
                    return -1;
                }

                tcl(0) = tcl_xd(0);
                tcl(1) = tcl_xd(1);
                tcl(2) = tcl_xd(2);
                cout<<"Num of frames to optimize extrisinc = "<<lidarPlanes.size()<<endl;
                cout<<"Final Tcl = "<<endl<<Rcl<<endl;
                cout<<tcl.transpose()<<endl;
                //
                string data_path;
                n.getParam("/calibration/path_Extrinsic", data_path);//根目录
                Rcl_=Rcl;
                Tcl_=tcl;
                //FileIO::WriteSenserParameters(Rcl,tcl,intrincMatrix,distParameter,data_path);
                vector<string> calibraion_res;
                calibraion_res.push_back( to_string(Rcl(0,0))+" "+to_string(Rcl(0,1))+" "+to_string(Rcl(0,2)) );
                calibraion_res.push_back( to_string(Rcl(1,0))+" "+to_string(Rcl(1,1))+" "+to_string(Rcl(1,2)) );
                calibraion_res.push_back( to_string(Rcl(2,0))+" "+to_string(Rcl(2,1))+" "+to_string(Rcl(2,2)) );
                calibraion_res.push_back( to_string(tcl[0])+" "+to_string(tcl[1])+" "+to_string(tcl[2]) );
                calibraion_res.push_back( to_string(intrincMatrix.at<double>(0,0))+" "+to_string(intrincMatrix.at<double>(0,1))+" "+to_string(intrincMatrix.at<double>(0,2)) );
                calibraion_res.push_back( to_string(intrincMatrix.at<double>(1,0))+" "+to_string(intrincMatrix.at<double>(1,1))+" "+to_string(intrincMatrix.at<double>(1,2)) );
                calibraion_res.push_back( to_string(intrincMatrix.at<double>(2,0))+" "+to_string(intrincMatrix.at<double>(2,1))+" "+to_string(intrincMatrix.at<double>(2,2)) );
                calibraion_res.push_back(to_string(distParameter.at<double>(0,0))+" "+
                            to_string(distParameter.at<double>(0,1))+" "+
                            to_string(distParameter.at<double>(0,2))+" "+
                            to_string(distParameter.at<double>(0,3))+" "+
                            to_string(distParameter.at<double>(0,4)));
                FileIO::WriteSting2Txt("/home/result/Extrinsic.txt",calibraion_res);

                vector<string> all_plane;
                for(int i=0;i<lidar_img_names.size();i++)
                {
                    if((cam_planes.find(i)!=cam_planes.end())&&(lidar_planes.find(i)!=lidar_planes.end()))
                    {
                        all_plane.push_back(to_string(lidar_planes[i][0](0))+" "+to_string(lidar_planes[i][0](1))+" "+to_string(lidar_planes[i][0](2))+" "+to_string(lidar_planes[i][0](3))+
                        " "+to_string(cam_planes[i][0](0))+" "+to_string(cam_planes[i][0](1))+" "+to_string(cam_planes[i][0](2))+" "+to_string(cam_planes[i][0](3)));
                        all_plane.push_back(to_string(lidar_planes[i][1](0))+" "+to_string(lidar_planes[i][1](1))+" "+to_string(lidar_planes[i][1](2))+" "+to_string(lidar_planes[i][1](3))+
                        " "+to_string(cam_planes[i][1](0))+" "+to_string(cam_planes[i][1](1))+" "+to_string(cam_planes[i][1](2))+" "+to_string(cam_planes[i][1](3)));
                        all_plane.push_back(to_string(lidar_planes[i][2](0))+" "+to_string(lidar_planes[i][2](1))+" "+to_string(lidar_planes[i][2](2))+" "+to_string(lidar_planes[i][2](3))+
                        " "+to_string(cam_planes[i][2](0))+" "+to_string(cam_planes[i][2](1))+" "+to_string(cam_planes[i][2](2))+" "+to_string(cam_planes[i][2](3)));
                    }
                }
                FileIO::WriteSting2Txt("/home/result/Planes.txt",all_plane);

            }//能够找到配对的激光和视觉平面
        }//外参标定结束
        else
            cout<<"Can not find matched lidar plane and camera plane"<<endl;
    }

    //lidar
    string path1,path2;
    path1="/home/result/Planes.txt";
    path2="/home/result/Extrinsic.txt";
    std::cout<<"start calculate error"<<std::endl;
    calculateExtrinsicError(path1,path2);
}