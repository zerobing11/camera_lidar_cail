#ifndef _Camera_
#define _Camera_

#include <stdio.h>    
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <assert.h>

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>    
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>    
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/miniflann.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include "ChessboradStruct.h"
#include "CornerDetAC.h"

#include "FileIO.h"
#include "CameraCalibrationNew.h"
#include "DrawImage.h"
#include "Util.h"

using namespace std;
using namespace cv;

namespace CAMERA
{
    struct BOARD
    {
        vector<cv::Point2f> corners,orderd_corners;
        vector<int> edge_indx,ordered_edge_indx;//4个元素
        Point2f centroid;
        int first_edge_indx;//决定了ordered_edge_indx的顺序
    
        //深拷贝
        BOARD operator=(BOARD& BOARDTmp)
        {
            corners = BOARDTmp.corners;
            orderd_corners = BOARDTmp.orderd_corners;
            edge_indx = BOARDTmp.edge_indx;
            ordered_edge_indx = BOARDTmp.ordered_edge_indx;
            centroid = BOARDTmp.centroid;
            first_edge_indx = BOARDTmp.first_edge_indx;
            return *this;
        };
    };

    struct CmpByValue {  
        bool operator()(const pair<int,float>& lhs, const pair<int,float>& rhs) {  
        return lhs.second > rhs.second;  
    }  
    };
    //升序
    struct CmpByValueAscend {  
        bool operator()(const pair<int,float>& lhs, const pair<int,float>& rhs) {  
        return lhs.second < rhs.second;  
    }  
    };

    struct MouseDate{
        cv::Mat org_img;
        int p[3][2];
        int click_count;
        bool isMouseCallback;
    };

    vector<cv::Point3d> ConvertToDouble(vector<cv::Point3f> data)
    {
        vector<cv::Point3d> res;
        for(int i=0;i<data.size();i++)
        {
            cv::Point3d temp;
            temp.x = data[i].x;
            temp.y  = data[i].y;
            temp.z  = data[i].z;
            res.push_back(temp);
        }
        return res;
    }

    vector<cv::Point2d> ConvertToDouble(vector<cv::Point2f> data)
    {
        vector<cv::Point2d> res;
        for(int i=0;i<data.size();i++)
        {
            cv::Point2d temp;
            temp.x = data[i].x;
            temp.y  = data[i].y;
            res.push_back(temp);
        }
        return res;
    }

    void on_mouse( int event, int x, int y, int flags, void* ustc)    
    {
        MouseDate* MD=(MouseDate*)ustc;
        cv::Mat img;

        if(MD->isMouseCallback)
        {
            if(event == cv::EVENT_LBUTTONDOWN)
            {
                MD->click_count++;
                if(MD->click_count==3)
                    MD->click_count = 0;
                cv::Point2f clickpoint = cv::Point2f((float)x,(float)y); 
                MD->p[MD->click_count][0]=x;
                MD->p[MD->click_count][1]=y;

                if(MD->click_count==0)
                    cv::circle( MD->org_img, clickpoint, 5,Scalar(255, 0, 0) ,cv::FILLED, cv::LINE_AA, 0 );
                if(MD->click_count==1)
                    cv::circle( MD->org_img, clickpoint, 5,Scalar(0, 255, 0) ,cv::FILLED, cv::LINE_AA, 0 );
                if(MD->click_count==2)
                    cv::circle( MD->org_img, clickpoint, 5,Scalar(0, 0, 255) ,cv::FILLED, cv::LINE_AA, 0 );
                cv::imshow("Image",MD->org_img);
            }
        }
    }
    
    Eigen::Vector4f Calculate_Planar_Model(vector<Point3f>& data,float th = 0.02)
    {
        pcl::PointCloud<pcl::PointXYZI> cloudin;
        for(int i=0;i<data.size();i++)
        {
            pcl::PointXYZI temp;
            temp.x = data[i].x;
            temp.y = data[i].y;
            temp.z = data[i].z;
            cloudin.points.push_back(temp);
        }
        
        std::vector<int> inliers;
        pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZI> (cloudin.makeShared()));
        pcl::RandomSampleConsensus<pcl::PointXYZI> ransac (model_p);
        ransac.setDistanceThreshold (th);
        ransac.computeModel();
        ransac.getInliers(inliers);
        Eigen::VectorXf planar_coefficients;
        ransac.getModelCoefficients(planar_coefficients);
        
        Eigen::Vector4f res(planar_coefficients(0),planar_coefficients(1),planar_coefficients(2),planar_coefficients(3));
        return res;
    }

    class Camera
    {
        private:
            //fingding box
            int min_x_th ;
            int min_y_th ;
            int max_x_th ;
            int max_y_th ;
            cv::Rect rect;

            int numofcorner;
            //阈值参数
            double corner_detect_threshold;  // 角点检测阈值
            double chessboard_threshold;     // 棋盘格识别阈值
            //图片信息
            bool initialization;//是否初始化
            int img_indx;
            cv::Size image_size,cornersize; 
            cv::Mat img,pre_img,org_img;
            vector<int> camera_cal_frame;//参与内参标定的图像序列号

            //角点信息
            bool isMouseCallback;
            bool detectfirstframe;
            int corner_height,corner_width;
            BOARD cur_boards[3],pre_boards[3];
            vector<int> reorder_info;
            vector<cv::Point2f> pre_first_edge_points;
            MouseDate MD;//鼠标事件传递变量
            map<int,vector<cv::Point2f>> all_corners;//存放用于标定点所有角点
            vector<vector<Point2f>> valid2d;//图像特征点坐标
		    vector<vector<Point3f>> valid3d;//特征点在世界坐标系下坐标
            vector<vector<float>> coor;

            vector<Point> corners_p;
            Corners corners_s;
            std::vector<cv::Mat> chessboards;

            //标定数据路径
            string path_root;
            string path_coor;
            vector<string> imgnames;
            string path_image;

            //calibration
            vector<cv::Mat> rotateMat;//旋转向量，后面会将其转换为矩阵 无论distparmater数据类型如何变换，输出的是double类型
		    vector<cv::Mat> translateMat;//平移向量，后面会将其转换为矩阵
            cv::Mat distParameter;//摄像机的畸变系数
		    cv::Mat intrincMatrix;//内参矩阵,矩阵大小是3×3,所有初始值是0，每个元素是1维的，数据类型是32位float;
            Mat mapx;
		    Mat mapy;
            map<int,vector<Eigen::Vector4f>> cam_planes;

        public:
            Camera(int height,int width,string path);//传入棋盘格尺寸和标定数据根目录
            Camera(int height,int width,string path,double corner_thresh,double chess_thresh);//传入棋盘格尺寸、标定数据根目录和阈值参数

            //
            bool choose_vaild_frame();//确定有效帧
            void sort_boards();//标定板排序

            //sort corners
            float euclideanDist(cv::Point2f& a, cv::Point2f& b);
            float DistToLine(cv::Point2f p,cv::Point2f a,cv::Point2f b);
            vector<int> SearchClosest(vector<cv::Point2f>& source_points,vector<cv::Point2f>& target_points);
            bool Update_Ordered_Info(int indx_bd);
            bool Ensure_ValidFrame(std::vector<cv::Mat> chessboards);
            void TrackCentroid();
            void DataClear();

            //fingding box update
            void Update_Rect_Th(cv::Rect r);
            bool Update_Rect(int x_max,int x_min,int y_max,int y_min,cv::Size img_size);

            //载入图片
            void init_img();//初始化
            bool add(string path);//载入标定数据

            //calibration
            void calibration();//内参标定
            void calibration(const cv::Mat& predefined_intrinsic, const cv::Mat& predefined_distortion);//使用预定义内参并计算外参
            void check();//标定数据检查
            void get_corners();//生成标定特征点
            
            //显示与数据接口
            void show();
            void GetIntrincMatrix(cv::Mat &intrincMatrix_);
            void GetDistParameter(cv::Mat &distParameter_);
            void GetPlanesModels(map<int,vector<Eigen::Vector4f>> &cam_planes_);//获取标定板平面模型
    };
        
    Camera::Camera(int height,int width,string path)
    {
        path_root=path;
        path_coor = path_root+"/3D.txt";
        path_image = path_root+"/names.txt";
        coor = FileIO::ReadTxt2Float(path_coor,false);
        imgnames = FileIO::ReadTxt2String(path_image,false);

        min_x_th =0;
        min_y_th =0;
        max_x_th =0;
        max_y_th =0;
        cv::Rect rect_(0, 0, 0, 0);
        rect=rect_;
        reorder_info={0,0,0};

        numofcorner=height*width;
        cornersize.height=width;
        cornersize.width=height;

        // 设置默认阈值
        corner_detect_threshold = 0.15;
        chessboard_threshold = 0.8;

        initialization=false;
        detectfirstframe = false;

        img_indx=-1;

        corner_height=height;
        corner_width=width;
        isMouseCallback = true;

        distParameter = Mat(1,5,CV_64FC1,Scalar::all(0));
    }

    Camera::Camera(int height,int width,string path,double corner_thresh,double chess_thresh)
    {
        path_root=path;
        path_coor = path_root+"/3D.txt";
        path_image = path_root+"/names.txt";
        coor = FileIO::ReadTxt2Float(path_coor,false);
        imgnames = FileIO::ReadTxt2String(path_image,false);

        min_x_th =0;
        min_y_th =0;
        max_x_th =0;
        max_y_th =0;
        cv::Rect rect_(0, 0, 0, 0);
        rect=rect_;
        reorder_info={0,0,0};

        numofcorner=height*width;
        cornersize.height=width;
        cornersize.width=height;

        // 设置自定义阈值
        corner_detect_threshold = corner_thresh;
        chessboard_threshold = chess_thresh;

        initialization=false;
        detectfirstframe = false;

        img_indx=-1;

        corner_height=height;
        corner_width=width;
        isMouseCallback = true;

        distParameter = Mat(1,5,CV_64FC1,Scalar::all(0));
    }

    bool Camera::Ensure_ValidFrame(std::vector<cv::Mat> chessboards)
    {
        // cout << "\n--- 帧有效性验证 ---" << endl;
        // cout << "期望的棋盘格数量: 3" << endl;
        // cout << "期望每个棋盘格的角点数: " << numofcorner << " (" << corner_height << "x" << corner_width << ")" << endl;
        // cout << "实际检测到的棋盘格数量: " << chessboards.size() << endl;
        
        if(chessboards.size() == 0) {
            // cout << "❌ 没有检测到任何棋盘格！" << endl;
            // cout << "可能原因：" << endl;
            // cout << "  1. 角点检测阈值太高，降低corner_detect_threshold" << endl;
            // cout << "  2. 棋盘格识别阈值太高，降低chessboard_threshold" << endl;
            // cout << "  3. 图像质量问题（模糊、光照等）" << endl;
            return false;
        }
        
        for(int i = 0; i < chessboards.size(); i++) {
            int actual_corners = chessboards[i].cols * chessboards[i].rows;
            // cout << "棋盘格" << i << ": " << chessboards[i].rows << "x" << chessboards[i].cols 
            //      << " = " << actual_corners << "个角点";
            if(actual_corners == numofcorner) {
                // cout << " ✓" << endl;
            } else {
                // cout << " ❌ (期望" << numofcorner << "个)" << endl;
            }
        }
        
        if((chessboards.size()==3)&&(chessboards[0].cols*chessboards[0].rows==numofcorner)&&
        (chessboards[1].cols*chessboards[1].rows==numofcorner)&&
        (chessboards[2].cols*chessboards[2].rows==numofcorner))
        {
            // cout << "✅ 帧验证通过！检测到3个有效的" << corner_height << "x" << corner_width << "棋盘格" << endl;
            return true;
        }
        else
        {
            // cout << "❌ 帧验证失败！" << endl;
            if(chessboards.size() != 3) {
                // cout << "问题：棋盘格数量不正确 (实际:" << chessboards.size() << ", 期望:3)" << endl;
                if(chessboards.size() < 3) {
                    // cout << "建议：降低chessboard_threshold或corner_detect_threshold" << endl;
                } else {
                    // cout << "建议：提高chessboard_threshold，减少误检" << endl;
                }
            }
            if(chessboards.size() >= 1 && chessboards[0].cols*chessboards[0].rows != numofcorner) {
                // cout << "问题：棋盘格尺寸不正确" << endl;
                // cout << "建议：检查棋盘格实际尺寸是否为" << corner_height << "x" << corner_width << endl;
            }
            return false;
        }
    }

    float Camera::euclideanDist(cv::Point2f& a, cv::Point2f& b)
    {
        cv::Point2f diff = a - b;
        return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
    }

    float Camera::DistToLine(cv::Point2f p,cv::Point2f a,cv::Point2f b)
    {
        float A,B,C;
        A = a.y-b.y;
        B = b.x-a.x;
        C = a.x*b.y-a.y*b.x;
        float dist = abs(A*p.x+B*p.y+C)/sqrt(A*A+B*B);
        return dist;
    }

    vector<int> Camera::SearchClosest(vector<cv::Point2f>& source_points,vector<cv::Point2f>& target_points)
    {
        // 检查输入是否为空
        if(source_points.empty() || target_points.empty())
        {
            std::cout << "SearchClosest函数输入的点为空" << std::endl;
        }
        
        //生成Mat对象
        cv::Mat source = cv::Mat(source_points).reshape(1);
        source.convertTo(source,CV_32F);

        cv::flann::KDTreeIndexParams indexParams(2); 
        cv::flann::Index kdtree(source, indexParams); //kd树索引建立完毕

        int quervNum = 1;
        vector<float> vecQuery(2);
        vector<int> vecIndex(quervNum),source_index;
        vector<float> vecDist(quervNum);
        cv::flann::SearchParams params(32);

        //计算最邻点
        for(int i=0;i<target_points.size();i++)
        {
            vecQuery.clear();
            vecQuery = { target_points[i].x, target_points[i].y};
            kdtree.knnSearch(vecQuery, vecIndex, vecDist, quervNum, params);
            source_index.push_back(vecIndex[0]);  
        }
        
        //查询是否有重复索引值
        vector<int> temp_index(source_index);
        sort(temp_index.begin(),temp_index.end());
        for(int i=1;i<temp_index.size();i++)
        {
        //cout<<temp_index[i]<<endl;
            if(temp_index[i-1]==temp_index[i])
                source_index.clear();
        }
            
        return source_index;
    }

    bool Camera::Update_Ordered_Info(int indx_bd)
    {
        //更新标定版上的ordered_edge_indx
        cur_boards[indx_bd].ordered_edge_indx.clear();
        vector<pair<int,float>> dist_map;
        Point2f firstpoint = cur_boards[indx_bd].corners[cur_boards[indx_bd].first_edge_indx];
        for(int i=0;i<4;i++)
        {
            Point2f tmppoint = cur_boards[indx_bd].corners[cur_boards[indx_bd].edge_indx[i]];
            float dist = euclideanDist(firstpoint,tmppoint);
            dist_map.push_back(pair<int,float>(cur_boards[indx_bd].edge_indx[i],dist));
        }
        sort(dist_map.begin(), dist_map.end(), CmpByValue()); 
        cur_boards[indx_bd].ordered_edge_indx.push_back(cur_boards[indx_bd].first_edge_indx);
        cur_boards[indx_bd].ordered_edge_indx.push_back(dist_map[2].first);
        cur_boards[indx_bd].ordered_edge_indx.push_back(dist_map[1].first);
        cur_boards[indx_bd].ordered_edge_indx.push_back(dist_map[0].first);//最远的距离
        
        //然后更新orderd_corners
        //根据最外侧的四个角点生成其他的24个角点
        Point2f A = cur_boards[indx_bd].corners[cur_boards[indx_bd].ordered_edge_indx[0]];
        Point2f B = cur_boards[indx_bd].corners[cur_boards[indx_bd].ordered_edge_indx[1]];
        Point2f C = cur_boards[indx_bd].corners[cur_boards[indx_bd].ordered_edge_indx[2]];
        Point2f D = cur_boards[indx_bd].corners[cur_boards[indx_bd].ordered_edge_indx[3]];
        Point2f vector1 = B-A;
        Point2f vector2 = D-C;
        if((vector2.x*vector1.x+vector2.y*vector1.y)<0)
        {
            Point2f tmp = C;
            C = D;
            D = tmp;
            int indx_tmp = cur_boards[indx_bd].ordered_edge_indx[3];
            cur_boards[indx_bd].ordered_edge_indx[3] = cur_boards[indx_bd].ordered_edge_indx[2];
            cur_boards[indx_bd].ordered_edge_indx[2] = indx_tmp;
        }
        //第一个点是白色，依次是红绿蓝
        cv::circle( img, A, 3,cv::Scalar(255,255,255) ,cv::FILLED, cv::LINE_AA, 0 );//红色
        cv::circle( img, B, 3,cv::Scalar(0,0,255) ,cv::FILLED, cv::LINE_AA, 0 );//红色
        cv::circle( img, C, 3,cv::Scalar(0,255,0) ,cv::FILLED, cv::LINE_AA, 0 );//绿色
        cv::circle( img, D, 3,cv::Scalar(255,0,0) ,cv::FILLED, cv::LINE_AA, 0 );//蓝色
        
        cur_boards[indx_bd].orderd_corners.clear();
        
        vector<pair<int,float>> start_dist_order,end_dist_order;
        for(int j=0;j<cur_boards[indx_bd].corners.size();j++)
        {
            float dist= DistToLine(cur_boards[indx_bd].corners[j],A,C);
            start_dist_order.push_back(pair<int,float>(j,dist));
            dist = DistToLine(cur_boards[indx_bd].corners[j],B,D);
            end_dist_order.push_back(pair<int,float>(j,dist));
        }
        sort(start_dist_order.begin(),start_dist_order.end(),CmpByValueAscend());//升序
        sort(end_dist_order.begin(),end_dist_order.end(),CmpByValueAscend());//升序
        
        vector<pair<int,float>> start_cd,end_cd;
        for(int i=0;i<cornersize.height;i++)
        {
            float dis = euclideanDist(A,cur_boards[indx_bd].corners[start_dist_order[i].first]);
            start_cd.push_back(pair<int,float>(start_dist_order[i].first,dis));
            dis = euclideanDist(B,cur_boards[indx_bd].corners[end_dist_order[i].first]);
            end_cd.push_back(pair<int,float>(end_dist_order[i].first,dis));
        }
        sort(start_cd.begin(),start_cd.end(),CmpByValueAscend());//升序
        sort(end_cd.begin(),end_cd.end(),CmpByValueAscend());//升序
        
        vector<Point2f> search_points;
        for(int row=0;row<cornersize.height;row++)
        {
            Point2f start = cur_boards[indx_bd].corners[start_cd[row].first];
            Point2f end = cur_boards[indx_bd].corners[end_cd[row].first];
            for(int col=0;col<cornersize.width;col++)
            {
                Point2f delta = (end-start)/(float)(cornersize.width-1);
                search_points.push_back(delta*col+start);
            }
        }

        //然后设置最近邻搜索
        vector<int> res = SearchClosest(cur_boards[indx_bd].corners,search_points);
        if(res.size()!=0)
        {
            cur_boards[indx_bd].orderd_corners.clear();
            for(int i=0;i<res.size();i++)
            {
                Point2f tmp = cur_boards[indx_bd].corners[res[i]];
                cur_boards[indx_bd].orderd_corners.push_back(tmp);
            }
        }
        else
        {
            return false;
        }
        
        return true;
    }

    void Camera::TrackCentroid()
    {
        reorder_info.clear();
        int indx3 = -1;
        float max_dist = -1;
        for(int i=0;i<3;i++)
        {
            if(cur_boards[i].centroid.y>max_dist)
            {
            max_dist = cur_boards[i].centroid.y;
            indx3 = i;
            }
        }
        reorder_info[2] = indx3;
        
        vector<int> rest;
        for(int i=0;i<3;i++)
        {
            if(i!=indx3)
            {
            rest.push_back(i);
            }
        }
        if(cur_boards[rest[0]].centroid.x>cur_boards[rest[1]].centroid.x)
        {
            reorder_info[0] = rest[0];
            reorder_info[1] = rest[1];
        }
        else
        {
            reorder_info[0] = rest[1];
            reorder_info[1] = rest[0];
        }
        struct BOARD board0 = cur_boards[reorder_info[0]];
        struct BOARD board1 = cur_boards[reorder_info[1]];
        struct BOARD board2 = cur_boards[reorder_info[2]];
        cur_boards[0] = board0;
        cur_boards[1] = board1;
        cur_boards[2] = board2;  
        
        
        vector<Point2f> curr_first_edge_points;
        vector<unsigned char> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(pre_img,img,pre_first_edge_points,curr_first_edge_points,status,error,cv::Size(21,21),5);
        
        for(int indx_bd=0;indx_bd<3;indx_bd++)
        {
            cv::circle( img, curr_first_edge_points[indx_bd], 6,cv::Scalar(28,255,255));
            float min_dist = 10000;
            int min_indx = -1;
            vector<pair<int,float>> map_dist;
            for(int i=0;i<4;i++)
            {
            float dist = euclideanDist(curr_first_edge_points[indx_bd],cur_boards[indx_bd].corners[cur_boards[indx_bd].edge_indx[i]]);
            map_dist.push_back(pair<int,float>(cur_boards[indx_bd].edge_indx[i],dist));
            }
            sort(map_dist.begin(),map_dist.end(),CmpByValue());//降序
            cur_boards[indx_bd].first_edge_indx = map_dist[3].first;
        }
        
        for(int indx_bd=0;indx_bd<3;indx_bd++)
        {
            pre_first_edge_points[indx_bd] = cur_boards[indx_bd].corners[cur_boards[indx_bd].first_edge_indx];
        }
    }

    bool Camera::choose_vaild_frame()
    {
        //遍历某个标定版
        int max_x = -100000;
        int min_x = 100000;
        int max_y = -100000;
        int min_y = 100000;
        //遍历当前帧，检测棋盘格并标注
        for (int indx_bd = 0; indx_bd < 3; indx_bd++)
        {
            cur_boards[indx_bd].corners.clear();//角点清零
            cur_boards[indx_bd].orderd_corners.clear();
            cur_boards[indx_bd].edge_indx.clear();
            cur_boards[indx_bd].ordered_edge_indx.clear();
            //更新标定板中的corner centroid、 edge_indx和centroid
            Point2f acc(0,0);
            //遍历棋盘格每个角点，对不同标定板的角点用不同颜色标注，并记录顶点序号
            for (int i = 0; i < chessboards[indx_bd].rows; i++)
            {
                for (int j = 0; j < chessboards[indx_bd].cols; j++)
                {
                    //获取角点序号、、、、
                    int d = chessboards[indx_bd].at<int>(i, j);
                    //更新标定板范围
                    cv::Point2f point(corners_s.p[d].x, corners_s.p[d].y);
                    if(max_x<(int)point.x)
                        max_x = point.x;
                    if(min_x>(int)point.x)
                        min_x = point.x;
                    if(max_y<(int)point.y)
                        max_y = point.y;
                    if(min_y>(int)point.y)
                        min_y = point.y;
                    cur_boards[indx_bd].corners.push_back(point);
                    //计算所有角点坐标和
                    acc  = acc + point;
                    //不同标定板角点标注
                    if(indx_bd==0)
                        cv::circle( img, point, 1,Scalar(255, 0, 0) ,cv::FILLED, cv::LINE_AA, 0 );
                    if(indx_bd==1)
                        cv::circle( img, point, 1,Scalar(0, 255, 0) ,cv::FILLED, cv::LINE_AA, 0 );
                    if(indx_bd==2)
                        cv::circle( img, point, 1,Scalar(0, 0, 255) ,cv::FILLED, cv::LINE_AA, 0 );
                    
                    //记录顶点序号
                    if((i==0)&&(j==0))
                        cur_boards[indx_bd].edge_indx.push_back(j+i*chessboards[indx_bd].cols);
                    if((i==0)&&(j==chessboards[indx_bd].cols-1))
                        cur_boards[indx_bd].edge_indx.push_back(j+i*chessboards[indx_bd].cols);
                    if((i==chessboards[indx_bd].rows-1)&&(j==0))
                        cur_boards[indx_bd].edge_indx.push_back(j+i*chessboards[indx_bd].cols);
                    if((i==chessboards[indx_bd].rows-1)&&(j==chessboards[indx_bd].cols-1))
                        cur_boards[indx_bd].edge_indx.push_back(j+i*chessboards[indx_bd].cols);
                }
            }  

            //标注标定板顶点
            for(int i=0;i<4;i++)
            {
                Point2f tmp = cur_boards[indx_bd].corners[cur_boards[indx_bd].edge_indx[i]];
                //cout<<"  Edge Point = "<<tmp.x<<" "<<tmp.y<<endl;
                if(indx_bd==0)
                    cv::circle( img, tmp, 3,Scalar(255, 0, 0) ,cv::FILLED, cv::LINE_AA, 0 );//蓝色
                if(indx_bd==1)
                    cv::circle( img, tmp, 3,Scalar(0, 255, 0) ,cv::FILLED, cv::LINE_AA, 0 );//绿色
                if(indx_bd==2)
                    cv::circle( img, tmp, 3,Scalar(0, 0, 255) ,cv::FILLED, cv::LINE_AA, 0 );//红色
            }
            //形心坐标
            Point2f centroid = acc/(float)numofcorner;
            cur_boards[indx_bd].centroid = centroid;
        }
        return Update_Rect(max_x,min_x,max_y,min_y,image_size);
    }

    void Camera::sort_boards()
    {
        if(detectfirstframe==false)
        {
            MD.org_img=img.clone();
            MD.isMouseCallback=true;
            MD.click_count=-1;
            MouseDate* pMD=&MD;
            cvNamedWindow("Image");
		    cvSetMouseCallback( "Image", on_mouse, (void*)pMD);

            cv::imshow ( "Image", img);
            //等待手动标注标定板顺序
            cv::waitKey(0);
            
            for(int j=0;j<3;j++)
            {
                cv::Point2f clickpoint = cv::Point2f((float)MD.p[j][0],(float)MD.p[j][1]); 
                cout<<MD.p[j][0]<<"  "<<MD.p[j][1]<<endl;
                float min_dist = 100000;
                int min_board_indx = -1;
                int min_corner_indx = -1;
                for(int indx_bd=0;indx_bd<3;indx_bd++)
                {
                    for(int i=0;i<4;i++)
                    {
                        int edge_indx = cur_boards[indx_bd].edge_indx[i];
                        float dist = euclideanDist(cur_boards[indx_bd].corners[edge_indx],clickpoint);
                        if(min_dist>dist)
                        {
                            min_board_indx = indx_bd;
                            min_corner_indx = edge_indx;
                            min_dist = dist;
                        }
                    }
                }//最近的点检测完毕
                cur_boards[min_board_indx].first_edge_indx = min_corner_indx;
                reorder_info[j] = min_board_indx;
                cout<<"min_board_indx = "<<min_board_indx<<" min_corner_indx = "<<min_corner_indx<<endl;
                cv::circle( img, cur_boards[min_board_indx].corners[min_corner_indx], 3,Scalar(255, 255, 255) ,cv::FILLED, cv::LINE_AA, 0 );
                //更新ordered_edge_indx和ordered_corner
                bool valid = Update_Ordered_Info(min_board_indx);//根据点的顺序重新对board进行排序
                if(valid)
                {
                    for(int i=0;i<cur_boards[min_board_indx].orderd_corners.size();i++)
                    {
                        Point2f tmp = cur_boards[min_board_indx].orderd_corners[i];
                        if((i==0)||(i==corner_height-1)||(i==corner_height*corner_width-corner_height)||(i==corner_height*corner_width-1))
                        cv::putText(img,std::to_string(i+j*numofcorner),tmp,cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(80,127,255),1,1);
                    }
                    cout<<"Update_Ordered_Info Success"<<endl;
                }
                else
                {
                cout<<"Update_Ordered_Info Fails"<<endl;
                }
                 
            }
            cv::imshow ( "Image", img);
            cv::waitKey(0);

            //初始化
            isMouseCallback = false;
            detectfirstframe = true;
            //最后交换顺序
            struct BOARD board0 = cur_boards[reorder_info[0]];
            struct BOARD board1 = cur_boards[reorder_info[1]];
            struct BOARD board2 = cur_boards[reorder_info[2]];
            cur_boards[0] = board0;
            cur_boards[1] = board1;
            cur_boards[2] = board2;
            
            pre_boards[0] = cur_boards[0];
            pre_boards[1] = cur_boards[1];
            pre_boards[2] = cur_boards[2];
            pre_first_edge_points.clear();
            for(int i=0;i<3;i++)
                pre_first_edge_points.push_back(cur_boards[i].corners[cur_boards[i].first_edge_indx]);
        }
        else
        {
            //根据上一个frame的重心来确定每个board中的first_edge_indx的序号
            TrackCentroid();
            Update_Ordered_Info(0);//根据点的顺序重新对board进行排序
            Update_Ordered_Info(1);//根据点的顺序重新对board进行排序
            Update_Ordered_Info(2);//根据点的顺序重新对board进行排序
            for(int indx_bd=0;indx_bd<3;indx_bd++)
            {
                for(int i=0;i<cur_boards[indx_bd].orderd_corners.size();i++)
                {
                    Point2f tmp = cur_boards[indx_bd].orderd_corners[i];
                    if((i==0)||(i==corner_height-1)||(i==corner_height*corner_width-corner_height)||(i==corner_height*corner_width-1))
                        cv::putText(img,std::to_string(i+indx_bd*numofcorner),tmp,cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(80,127,255),1,1);
                }
            }
            cv::imshow ( "Image", img);
            cv::waitKey(50);  // 延长到100ms，让边框检测结果更清楚
            //最后交换顺序
            struct BOARD board0 = cur_boards[0];
            struct BOARD board1 = cur_boards[1];
            struct BOARD board2 = cur_boards[2];
            cur_boards[0] = board0;
            cur_boards[1] = board1;
            cur_boards[2] = board2;
            
            pre_boards[0] = cur_boards[0];
            pre_boards[1] = cur_boards[1];
            pre_boards[2] = cur_boards[2];

        }
    }

    void Camera::Update_Rect_Th(cv::Rect r)
    {
        min_x_th = r.x;
        min_y_th = r.y;
        max_x_th = r.width+r.x;
        max_y_th = r.height+r.y;
    }

    bool Camera::Update_Rect(int x_max,int x_min,int y_max,int y_min,cv::Size img_size)
    {
        bool res = false;
        if((rect.x==0)&&(rect.y==0)&&(rect.width==0)&&(rect.width==0))
        {
            rect.x = x_min;
            rect.y = y_min;
            rect.width = x_max - x_min;
            rect.height = y_max - y_min;
            res = true;
        }
        else
        {
            int max_x_org = rect.x+rect.width;
            int max_y_org = rect.y+rect.height;
            if(x_min<rect.x)
            {
                rect.x = x_min;
                if((min_x_th-rect.x)>0.01*img_size.width)
                {
                    res = true;
                }
            }
            if(y_min<rect.y)
            {
                rect.y = y_min;
                if((min_y_th-rect.y)>0.01*img_size.height)
                    res = true;
            }
            
            if(max_x_org<x_max)
            {
                rect.width = x_max-rect.x;
                if((x_max-max_x_th)>0.01*img_size.width)
                    res = true;
            }
            else
            rect.width = max_x_org-rect.x;

            
            if(max_y_org<y_max)
            {
                rect.height = y_max-rect.y;
                if((y_max-max_y_th)>0.01*img_size.height)
                    res  = true;
            }
            else
                rect.height = max_y_org-rect.y;
        }
        
        if(res)
            Update_Rect_Th(rect);
        return res;
    }

    void Camera::init_img()
    {
        initialization=true;
        cout<<"camera_calibration Start!"<<endl;
        cout<<"Sort Chessboard corners"<<endl;

        image_size.width = img.cols;
        image_size.height = img.rows;   
    }
    
    void Camera::DataClear()
    {
        chessboards.clear();
        corners_s.p.clear();
        corners_s.v1.clear();
        corners_s.v2.clear();
        corners_s.score.clear();
    }


    bool Camera::add(string path)
    {
        // 读取原始图像（彩色模式）
        org_img=cv::imread(path,1);
        // 克隆图像用于处理，避免修改原始图像
        img = org_img.clone();
        // 增加图像索引计数器
        img_indx++;
        // 在图像上绘制当前帧编号，用于可视化显示
        cv::putText(img,std::to_string(img_indx),cv::Point2f(10,30),cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(155,155,155),3);
        
        // 如果是第一帧，进行初始化
        if(!initialization)
            init_img();
            
        // 清除之前的数据
        DataClear();
        
        // 创建角点检测器和棋盘格结构体
        CornerDetAC corner_detector(img);
        ChessboradStruct chessboardstruct;

        // 检测棋盘格角点，使用配置的阈值
        // corners_p: 输出参数，存储检测到的角点像素坐标；
        // corners_s: 输出参数，存储角点的分数/质量信息
         cout << "===  帧" << img_indx << " ===" << endl;
        // cout << "角点检测阈值: " << corner_detect_threshold << endl;
        
        corner_detector.detectCorners(img, corners_p, corners_s, corner_detect_threshold);
        
        // 调试信息：角点检测结果
         cout << "检测到的角点数量: " << corners_s.p.size() << endl;
        // cout << "角点质量分数数量: " << corners_s.score.size() << endl;
        // if(corners_s.p.size() > 0) {
        //     cout << "角点质量分数范围: [" << *min_element(corners_s.score.begin(), corners_s.score.end()) 
        //          << ", " << *max_element(corners_s.score.begin(), corners_s.score.end()) << "]" << endl;
        // }

 //由此往下是可视化。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        // 可视化角点并保存图像
        cv::Mat corner_vis_img = org_img.clone();
        for(int i = 0; i < corners_s.p.size(); i++) {
            cv::Point2f corner_point = corners_s.p[i];
            // 根据角点质量分数设置颜色：红色(低质量) -> 黄色(中等) -> 绿色(高质量)
            float score = corners_s.score[i];
            cv::Scalar color;
            if(score < 0.1) {
                color = cv::Scalar(0, 0, 255); // 红色 - 低质量
            } else if(score < 0.15) {
                color = cv::Scalar(0, 165, 255); // 橙色 - 中等质量
            } else {
                color = cv::Scalar(0, 255, 0); // 绿色 - 高质量
            }
            
            // 画角点圆圈
            cv::circle(corner_vis_img, corner_point, 6, color, 2);
            // 标注角点序号
            cv::putText(corner_vis_img, std::to_string(i), 
                       cv::Point2f(corner_point.x + 8, corner_point.y - 8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
        
        // 创建保存目录
        string corner_debug_dir = path_root + "/img_corner_test";
        string mkdir_cmd = "mkdir -p " + corner_debug_dir;
        system(mkdir_cmd.c_str());
        
        // 保存角点可视化图像
        string corner_save_path = corner_debug_dir + "/frame_" + 
                                 std::to_string(img_indx) + "_corners.png";
        cv::imwrite(corner_save_path, corner_vis_img);
        cout << "角点可视化图像已保存到: " << corner_save_path << endl;
//由此往上是可视化。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。

        ImageChessesStruct ics;
        chessboardstruct.chessboardsFromCorners(corners_s, chessboards, chessboard_threshold);
        
        // 调试信息：棋盘格识别结果
         cout << "识别到的棋盘格数量: " << chessboards.size() << endl;


//由此往下是可视化。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        // 可视化棋盘格识别结果
        cv::Mat chessboard_vis_img = org_img.clone();
        vector<cv::Scalar> board_colors = {
            cv::Scalar(255, 0, 0),   // 蓝色
            cv::Scalar(0, 255, 0),   // 绿色
            cv::Scalar(0, 0, 255),   // 红色
            cv::Scalar(255, 255, 0), // 青色
            cv::Scalar(255, 0, 255), // 品红色
        };

        for(int board_idx = 0; board_idx < chessboards.size() && board_idx < 5; board_idx++)
        {
            cv::Scalar color = board_colors[board_idx];
            cv::Mat& board = chessboards[board_idx];

            cout << "绘制棋盘格" << board_idx << "的角点分布..." << endl;

            for(int row = 0; row < board.rows; row++)
            {
                for(int col = 0; col < board.cols; col++)
                {
                    int corner_idx = board.at<int>(row, col);
                    if(corner_idx >= 0 && corner_idx < corners_s.p.size())
                    {
                        cv::Point2f corner_point = corners_s.p[corner_idx];

                        // 画角点圆圈
                        cv::circle(chessboard_vis_img, corner_point, 8, color, 3);

                        // 标注棋盘格内的位置(row,col)
                        string pos_label = "(" + std::to_string(row) + "," + std::to_string(col) + ")";
                        cv::putText(chessboard_vis_img, pos_label,
                                   cv::Point2f(corner_point.x + 10, corner_point.y - 10),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);

                        // 画连接线构成网格
                        if(col < board.cols - 1) // 水平连线
                        {
                            int next_corner_idx = board.at<int>(row, col + 1);
                            if(next_corner_idx >= 0 && next_corner_idx < corners_s.p.size())
                            {
                                cv::Point2f next_corner = corners_s.p[next_corner_idx];
                                cv::line(chessboard_vis_img, corner_point, next_corner, color, 2);
                            }
                        }
                        if(row < board.rows - 1) // 垂直连线
                        {
                            int next_corner_idx = board.at<int>(row + 1, col);
                            if(next_corner_idx >= 0 && next_corner_idx < corners_s.p.size())
                            {
                                cv::Point2f next_corner = corners_s.p[next_corner_idx];
                                cv::line(chessboard_vis_img, corner_point, next_corner, color, 2);
                            }
                        }
                    }
                }
            }

            // 在棋盘格左上角标注棋盘格编号和尺寸
            if(board.rows > 0 && board.cols > 0)
            {
                int first_corner_idx = board.at<int>(0, 0);
                if(first_corner_idx >= 0 && first_corner_idx < corners_s.p.size())
                {
                    cv::Point2f first_corner = corners_s.p[first_corner_idx];
                    string board_label = "Board" + std::to_string(board_idx) +
                                        "(" + std::to_string(board.rows) + "x" + std::to_string(board.cols) + ")";
                    cv::putText(chessboard_vis_img, board_label,
                               cv::Point2f(first_corner.x - 50, first_corner.y - 20),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                }
            }
        }

        // 标记未被使用的角点
        vector<bool> used_corners(corners_s.p.size(), false);
        for(int board_idx = 0; board_idx < chessboards.size(); board_idx++)
        {
            cv::Mat& board = chessboards[board_idx];
            for(int row = 0; row < board.rows; row++)
            {
                for(int col = 0; col < board.cols; col++)
                {
                    int corner_idx = board.at<int>(row, col);
                    if(corner_idx >= 0 && corner_idx < corners_s.p.size())
                    {
                        used_corners[corner_idx] = true;
                    }
                }
            }
        }

        // 绘制未被使用的角点
        int unused_count = 0;
        for(int i = 0; i < corners_s.p.size(); i++)
        {
            if(!used_corners[i])
            {
                cv::Point2f corner_point = corners_s.p[i];
                // 用灰色圆圈标记未使用的角点
                cv::circle(chessboard_vis_img, corner_point, 6, cv::Scalar(128, 128, 128), 2);
                // 标注为未使用
                cv::putText(chessboard_vis_img, "unused",
                           cv::Point2f(corner_point.x + 10, corner_point.y + 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(128, 128, 128), 1);
                unused_count++;
            }
        }
        cout << "未被使用的角点数量: " << unused_count << endl;

        // 保存棋盘格可视化图像
        string chessboard_save_path = corner_debug_dir + "/frame_" +
                                     std::to_string(img_indx) + "_chessboards.png";
        cv::imwrite(chessboard_save_path, chessboard_vis_img);
        cout << "棋盘格可视化图像已保存到: " << chessboard_save_path << endl;

//由此往上是可视化。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。

        bool ischoose = false; // 标记当前帧是否被选中用于标定
        
        // 检查当前帧是否为有效帧（角点数量正确）
        if(Ensure_ValidFrame(chessboards))
        {
            // 选择有效帧，判断该帧是否适合用于标定
            ischoose =choose_vaild_frame();
            
            // 根据选择结果绘制不同颜色的矩形框
            if(ischoose)
                cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2); // 绿色：被选中
            else
                cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2); // 红色：未被选中
                
            // 如果是第一帧，调整标定板顺序
            sort_boards();
            
            // 收集当前帧所有三个标定板的角点
            vector<cv::Point2f> three_bd_corners;
            for(int indx_bd = 0;indx_bd<3;indx_bd++)
            {
                for(int i=0;i<pre_boards[indx_bd].orderd_corners.size();i++)
                    three_bd_corners.push_back(pre_boards[indx_bd].orderd_corners[i]);
            }
            
            // 如果该帧被选中，保存角点信息用于标定
            if(ischoose)
            {
                // 将当前帧的角点信息存储到全局角点映射中
                all_corners.insert(pair<int,vector<Point2f>>(img_indx,three_bd_corners));
                // 将当前帧索引添加到标定帧列表中
                camera_cal_frame.push_back(img_indx);
            }
        }
        else
        {
            // 如果检测到当前帧的角点数量错误（无效帧）
            if(detectfirstframe)
            {
                // 使用光流法跟踪第一帧的边缘点
                vector<Point2f> curr_first_edge_points;
                vector<unsigned char> status;  // 跟踪状态
                vector<float> error;           // 跟踪误差
                
                // 使用金字塔LK光流算法跟踪特征点
                cv::calcOpticalFlowPyrLK(pre_img,img,pre_first_edge_points,curr_first_edge_points,status,error,cv::Size(21,21),5);
                
                // 更新跟踪点
                pre_first_edge_points = curr_first_edge_points;
                
                // 在图像上绘制跟踪到的点（黄色圆圈）
                for(int i=0;i<3;i++)
                {
                    cv::circle( img, pre_first_edge_points[i], 6,cv::Scalar(28,255,255));
                }
                
                // 绘制白色矩形框表示无效帧
                cv::rectangle(img, rect, cv::Scalar(255, 255, 255), 2);
                
                // 显示处理后的图像
                cv::imshow ( "Image", img);
                cv::waitKey(500);
            }
        }
        
        // 保存当前帧作为下一帧的前一帧，用于光流跟踪
        pre_img = img.clone();
        
        // 返回该帧是否被选中用于标定
        return ischoose;
    }

    void Camera::check()
    {
        cout<<"numbers of input images:"<<img_indx+1<<endl;
        cout<<"numbers of frame will be used to calibration:"<<camera_cal_frame.size()<<endl<<endl;
        if(camera_cal_frame.size()<2)
		{
			cout<<"camera_cal_frame:"<<camera_cal_frame.size()<<endl;
			cout<<"no enough images can be used to calibration!"<<endl;
			abort();
		}
    }

    void Camera::get_corners()
    {
        for(int p=0;p<camera_cal_frame.size();p++)
		{
			vector<Point2f> temp;
			vector<Point3f> temp3d;
			//将标定特征点存入valid2d
			for(int i=0;i<all_corners[camera_cal_frame[p]].size();i++)
			{
				temp.push_back(all_corners[camera_cal_frame[p]][i]);
			}
			valid2d.push_back(temp);
			temp.clear();
			
			for(int i=0;i<coor.size();i++)
			{
				temp3d.push_back(Point3f(coor[i][0],coor[i][1],coor[i][2]));
			}
			valid3d.push_back(temp3d);
			temp3d.clear();  
        }

    }

    void Camera::calibration()
    {
        //check
        check();
        get_corners();
        
        cout<<"Start calculate camera intrincMatrix!!!"<<endl;
        intrincMatrix = CameraCalibration3D::initCameraMatrix2DNew(valid3d,valid2d,image_size,true);
        cout<<"Initial intrincMatrix = "<<endl;
		cout<<intrincMatrix<<endl;
		cout<<"Initial distParameter = "<<endl;
		cout<<distParameter<<endl<<endl;

        cv::calibrateCamera(valid3d,valid2d,image_size,intrincMatrix,distParameter,rotateMat,translateMat,CALIB_USE_INTRINSIC_GUESS);//cv::CALIB_USE_INTRINSIC_GUESS
        cout<<"Final intrincMatrix = "<<endl;
		cout<<intrincMatrix<<endl;
		cout<<"Final distParameter = "<<endl;
            cout<<distParameter<<endl;

		cv::initUndistortRectifyMap(intrincMatrix,distParameter,Mat::eye(3,3,CV_32FC1),intrincMatrix,image_size,CV_32FC1,mapx,mapy);
        cout<<"Camera calibration has finished"<<endl<<endl;
    }

    void Camera::calibration(const cv::Mat& predefined_intrinsic, const cv::Mat& predefined_distortion)
    {
        //检查数据并获取角点
        check();
        get_corners();
        
        //使用预定义的内参和畸变参数
        cout<<"Using predefined camera intrinsic parameters:"<<endl;
        intrincMatrix = predefined_intrinsic.clone();
        distParameter = predefined_distortion.clone();
        
        cout<<"Predefined intrincMatrix = "<<endl;
		cout<<intrincMatrix<<endl;
		cout<<"Predefined distParameter = "<<endl;
		cout<<distParameter<<endl<<endl;

        // rotateMat.clear();
        // translateMat.clear();
        
        cout<<"Start calculating camera extrinsic parameters using PnP..."<<endl;
        
        //对每个有效帧使用PnP算法计算外参
        for(int i = 0; i < valid2d.size(); i++)
        {
            cv::Mat rvec, tvec; //旋转向量和平移向量
            
            //使用solvePnP计算当前帧的外参
            bool success = cv::solvePnP(valid3d[i], valid2d[i], intrincMatrix, distParameter, 
                                      rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            
            if(success)
            {
                //将旋转向量转换为旋转矩阵
                cv::Mat rmat;
                cv::Rodrigues(rvec, rmat);
                
                //存储外参
                rotateMat.push_back(rmat.clone());
                translateMat.push_back(tvec.clone());
                
                cout<<"Frame "<<i<<" - Rotation matrix:"<<endl<<rmat<<endl;
                cout<<"Frame "<<i<<" - Translation vector:"<<endl<<tvec<<endl;
            }
            else
            {
                cout<<"Failed to calculate extrinsic parameters for frame "<<i<<endl;
            }
        }

        //初始化去畸变映射
		cv::initUndistortRectifyMap(intrincMatrix,distParameter,Mat::eye(3,3,CV_32FC1),intrincMatrix,image_size,CV_32FC1,mapx,mapy);
        
        cout<<"Extrinsic parameter calculation has finished!"<<endl;
        cout<<"Total valid frames: "<<rotateMat.size()<<endl<<endl;
    }

    void Camera::show()
    {
        cout<<"Starting intrinsic and extrinsic parameter validation..."<<endl;
        
        // 创建零畸变参数矩阵，用于对比投影效果
        cv::Mat distParameter_zero = Mat(1,5,CV_64FC1,Scalar::all(0));
        
        // 遍历所有用于标定的图像帧
        for(int p=0;p<camera_cal_frame.size();p++)
		{
			// 获取当前帧的图像ID和文件名
			int id_img = camera_cal_frame[p];
			vector<string> name;
			name = read_format(imgnames[id_img]," ");
			
			cout<<"Processing frame "<<p+1<<"/"<<camera_cal_frame.size()<<" (image: "<<name[1]<<")"<<endl;
			
			// 读取原始图像
			Mat img_org =cv::imread(path_root+"/leftImg/left_"+name[1]+".png",1);
			if(img_org.empty())
			{
				cout<<"Error: Cannot load image "<<name[1]<<endl;
				continue;
			}
			
			Mat img_unditort;  // 去畸变后的图像
			
			// 使用内参和畸变参数进行去畸变
			cv::remap(img_org,img_unditort,mapx,mapy,INTER_LINEAR);
			
			// 使用计算得到的外参将3D点投影到2D图像
			vector<Point2d> projected_points_distort;   // 投影点（考虑畸变）
			vector<Point2d> projected_points_undistort; // 投影点（不考虑畸变）
			
			// 使用当前帧的外参进行投影（考虑畸变）
			cv::projectPoints(ConvertToDouble(valid3d[p]), rotateMat[p], translateMat[p], 
			                 intrincMatrix, distParameter, projected_points_distort);
			
			// 使用当前帧的外参进行投影（不考虑畸变）
			cv::projectPoints(ConvertToDouble(valid3d[p]), rotateMat[p], translateMat[p], 
			                 intrincMatrix, distParameter_zero, projected_points_undistort);
			
			// 创建显示图像
			Mat img_with_points = img_org.clone();
			Mat img_undistort_with_points = img_unditort.clone();
			
			// 计算投影误差
			double total_error = 0.0;
			int point_count = 0;
			
			// 在图像上绘制检测到的角点和投影角点
			for(int j=0;j<projected_points_distort.size() && j<valid2d[p].size();j++)
			{
				// 绘制投影点（蓝色圆圈）
				cv::circle(img_with_points, projected_points_distort[j], 4, Scalar(255, 0, 0), 2);
				cv::circle(img_undistort_with_points, projected_points_undistort[j], 4, Scalar(255, 0, 0), 2);
				
				// 绘制检测到的角点（红色圆圈）
				cv::circle(img_with_points, valid2d[p][j], 3, Scalar(0, 0, 255), -1);
				
				// 绘制连接线（绿色）
				cv::line(img_with_points, valid2d[p][j], projected_points_distort[j], Scalar(0, 255, 0), 1);
				
				// 计算重投影误差
				double dx = valid2d[p][j].x - projected_points_distort[j].x;
				double dy = valid2d[p][j].y - projected_points_distort[j].y;
				double error = sqrt(dx*dx + dy*dy);
				total_error += error;
				point_count++;
			}
			
			// 计算平均重投影误差
			double avg_error = (point_count > 0) ? total_error / point_count : 0.0;
			cout<<"Frame "<<p<<" - Average reprojection error: "<<avg_error<<" pixels"<<endl;
			
			// 在去畸变图像上绘制标定边框（绿色矩形）
			cv::rectangle(img_undistort_with_points, rect, cv::Scalar(0, 255, 0), 2);
			
			// 水平拼接原始图像和去畸变图像
			cv::Mat merge_img_final;
			Util::imageJoinHorizon(img_with_points, img_undistort_with_points, merge_img_final);
			
			// 添加文本信息
			string info_text = "Frame " + std::to_string(p+1) + "/" + std::to_string(camera_cal_frame.size());
			string error_text = "Avg Error: " + std::to_string(avg_error).substr(0,5) + " px";
			string legend_text = "Blue: Projected | Red: Detected | Green: Connection";
			
			cv::putText(merge_img_final, info_text, cv::Point2f(10,30), 
			           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);
			cv::putText(merge_img_final, error_text, cv::Point2f(10,60), 
			           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
			cv::putText(merge_img_final, legend_text, cv::Point2f(10,90), 
			           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
			
			// 显示拼接后的对比图像
			cv::imshow("Intrinsic & Extrinsic Validation", merge_img_final);
			
			//等待按键继续（ESC退出，其他键继续）
			int key = cv::waitKey(0);
			if(key == 27) // ESC键
			{
				cout<<"Validation stopped by user."<<endl;
				break;
			}
		}
		
		cout<<"Parameter validation completed!"<<endl;
    }

    void Camera::GetIntrincMatrix(cv::Mat &intrincMatrix_)
    {
        intrincMatrix_=intrincMatrix.clone();
    }

    void Camera::GetDistParameter(cv::Mat &distParameter_)
    {
        distParameter_=distParameter.clone();
    }

    /**
     * @brief 计算相机坐标系下的平面模型参数
     * 
     * 该函数用于计算每个标定图像中三个标定板在相机坐标系下的平面方程参数。
     * 平面方程形式：ax + by + cz + d = 0，其中(a,b,c)是法向量，d是距离参数。
     * 
     * 功能：
     * 1. 对每帧标定图像，计算三个标定板的平面参数
     * 2. 使用两种方法计算平面参数进行对比验证
     * 3. 将世界坐标系下的平面转换到相机坐标系
     * 4. 输出平面参数用于后续的相机-激光雷达标定
     * 
     * @param cam_planes_ 输出参数，存储每帧图像中三个标定板的平面参数
     */
    void Camera::GetPlanesModels(map<int,vector<Eigen::Vector4f>> &cam_planes_)
    {
        // 遍历所有标定图像的角点数据
        for(map<int,vector<Point2f>>::iterator iter = all_corners.begin();iter!=all_corners.end();iter++)
		{
			// 对图像角点进行去畸变处理
			vector<cv::Point2f> undistorpoints;
			cv::undistortPoints(iter->second, undistorpoints, intrincMatrix, distParameter,cv::noArray(),intrincMatrix);
			
			// 使用PnP算法求解相机位姿（旋转和平移）
            cv::Mat angleaxis;  // 旋转向量
			cv::Mat cvt;        // 平移向量
			
			// 先用EPNP算法进行初估计
			cv::solvePnP ( ConvertToDouble(valid3d[0]), ConvertToDouble(undistorpoints), intrincMatrix, 
			              Mat(1,5,CV_64FC1,Scalar::all(0)), angleaxis, cvt, false ,cv::SOLVEPNP_EPNP);
			
			// 再用迭代算法进行精确求解
			cv::solvePnP ( ConvertToDouble(valid3d[0]), ConvertToDouble(undistorpoints), intrincMatrix, 
			              Mat(1,5,CV_64FC1,Scalar::all(0)), angleaxis, cvt, true ,cv::SOLVEPNP_ITERATIVE);
			
			// 将旋转向量转换为旋转矩阵
			cv::Mat cvR;
			cv::Rodrigues ( angleaxis, cvR );
			
			// 输出相机位姿信息用于调试
			cout<<"cvR:"<<cvR<<endl;
			cout<<"cvt:"<<cvt<<endl;
			
			// 将OpenCV的旋转矩阵转换为Eigen格式
            Eigen::Matrix3f R_eigen;  // 相机到世界的旋转矩阵
			Eigen::Vector3f t_eigen;  // 相机到世界的平移向量
			R_eigen<<(float)cvR.at<double>(0,0),(float)cvR.at<double>(0,1),(float)cvR.at<double>(0,2),
				(float)cvR.at<double>(1,0),(float)cvR.at<double>(1,1),(float)cvR.at<double>(1,2),
				(float)cvR.at<double>(2,0),(float)cvR.at<double>(2,1),(float)cvR.at<double>(2,2);
			t_eigen<<(float)cvt.at<double>(0),(float)cvt.at<double>(1),(float)cvt.at<double>(2);
			
			// 存储两种方法计算的平面参数
			vector<Eigen::Vector4f> tmp_Plane,tmp_Plane2;
            vector<Eigen::Vector4f> world_plane;  // 世界坐标系下的平面参数
            vector<Eigen::Vector3f> world_centroid; // 世界坐标系下的平面中心点
            vector<cv::Point3f> valid3d_plane;
            
            // ========== 方法1：基于3D角点直接计算平面参数 ==========
            for(int indx_bd = 0;indx_bd<3;indx_bd++)
            {
                // 计算当前标定板的中心点坐标
                float acc_x = 0, acc_y = 0, acc_z = 0;
                vector<Point3f> one_board;
                
                // 收集当前标定板的所有角点
                for(int i=0;i<numofcorner;i++)
                {
                    one_board.push_back(valid3d[0][i+indx_bd*numofcorner]);
                    acc_x += valid3d[0][i+indx_bd*numofcorner].x;
                    acc_y += valid3d[0][i+indx_bd*numofcorner].y;
                    acc_z += valid3d[0][i+indx_bd*numofcorner].z;
                }
                
                // 计算标定板中心点
                acc_x /= numofcorner;
                acc_y /= numofcorner;
                acc_z /= numofcorner;
                world_centroid.push_back(Eigen::Vector3f(acc_x,acc_y,acc_z));
                
                // 计算平面方程参数 (ax + by + cz + d = 0)
                Eigen::Vector4f plane_model = Calculate_Planar_Model(one_board);
                world_plane.push_back(plane_model);
                cout<<"world plane = "<<plane_model.transpose()<<endl;
            }
            
            // 将世界坐标系下的平面转换到相机坐标系
            for(int i=0;i<3;i++)
			{
                // 获取平面法向量
                Eigen::Vector3f normal(world_plane[i][0],world_plane[i][1],world_plane[i][2]);
                
                // 确保法向量方向一致（d > 0）
                if(world_plane[i][3]<0)
                {
                    normal = -normal;
                    world_plane[i][0]=-world_plane[i][0];
                    world_plane[i][1]=-world_plane[i][1];
                    world_plane[i][2]=-world_plane[i][2];
                    world_plane[i][3]=-world_plane[i][3];
                }
                
                // 将世界坐标系法向量转换到相机坐标系
				Eigen::Vector3f cam_normal = R_eigen*normal;
                
                // 计算相机坐标系下的距离参数
                float cam_D = world_plane[i][3]-(cam_normal[0]*t_eigen[0]+cam_normal[1]*t_eigen[1]+cam_normal[2]*t_eigen[2]);
                
                // 确保法向量指向相机
				if((R_eigen*world_centroid[i]+t_eigen).adjoint()*cam_normal>0)
                {
                    cam_normal = -cam_normal;
                    cam_D=-cam_D;
                }
                
                // 存储相机坐标系下的平面参数
				tmp_Plane.push_back(Eigen::Vector4f(cam_normal[0],cam_normal[1],cam_normal[2],cam_D));
			}
            
            // ========== 方法2：基于标准平面模型重新计算 ==========
            // 创建标准的3D平面点（6x4网格，间距0.155m）
    //         for(int i=0;i<6;i++)
    //         {
    //             for(int j=0;j<4;j++)
    //             {
    //                 valid3d_plane.push_back(cv::Point3f(-j*0.155,-i*0.155,0));
    //             }
    //         }
    //
    //         vector<cv::Point2f> p;
    //         Eigen::Vector3f normal(0,0,-1);  // 标准平面的法向量
    //
    //         // 对每个标定板重新计算平面参数
    //         for(int i=0;i<3;i++)
    //         {
    //             // 收集当前标定板的角点
    //             for(int j=0;j<numofcorner;j++)
    //             {
    //                 p.push_back(undistorpoints[j+i*numofcorner]);
    //             }
    //
    //             // 重新求解PnP
    //             cv::solvePnP ( ConvertToDouble(valid3d_plane), ConvertToDouble(p), intrincMatrix,
    //                           Mat(1,5,CV_64FC1,Scalar::all(0)), angleaxis, cvt, false ,cv::SOLVEPNP_EPNP);
			 //    cv::solvePnP ( ConvertToDouble(valid3d_plane), ConvertToDouble(p), intrincMatrix,
    //                           Mat(1,5,CV_64FC1,Scalar::all(0)), angleaxis, cvt, true ,cv::SOLVEPNP_ITERATIVE);
    //
    //             // 更新旋转矩阵
    //             cv::Rodrigues ( angleaxis, cvR );
    //             R_eigen<<(float)cvR.at<double>(0,0),(float)cvR.at<double>(0,1),(float)cvR.at<double>(0,2),
				// (float)cvR.at<double>(1,0),(float)cvR.at<double>(1,1),(float)cvR.at<double>(1,2),
				// (float)cvR.at<double>(2,0),(float)cvR.at<double>(2,1),(float)cvR.at<double>(2,2);
			 //    t_eigen<<(float)cvt.at<double>(0),(float)cvt.at<double>(1),(float)cvt.at<double>(2);
    //
    //             cout<<"cvR:"<<cvR<<endl;
			 //    cout<<"cvt:"<<cvt<<endl;
    //
    //             // 计算相机坐标系下的平面参数
    //             normal=Eigen::Vector3f(0,0,-1);
    //             Eigen::Vector3f cam_normal = R_eigen*normal;
    //             float cam_D = -(cam_normal[0]*t_eigen[0]+cam_normal[1]*t_eigen[1]+cam_normal[2]*t_eigen[2]);
    //
    //             // 确保法向量指向相机
				// if((t_eigen).adjoint()*cam_normal>0)
    //             {
    //                 cam_normal = -cam_normal;
    //                 cam_D=-cam_D;
    //             }
    //
    //             // 存储第二种方法的平面参数
				// tmp_Plane2.push_back(Eigen::Vector4f(cam_normal[0],cam_normal[1],cam_normal[2],cam_D));
    //             p.clear();
    //         }
            
            // 输出两种方法的对比结果
            cout<<"method 1 plane param = "<<tmp_Plane[0].transpose()<<" "<<tmp_Plane[1].transpose()<<" "<<tmp_Plane[2].transpose()<<endl<<endl;
            // cout<<"method 2 plane param = "<<tmp_Plane2[0].transpose()<<" "<<tmp_Plane2[1].transpose()<<" "<<tmp_Plane2[2].transpose()<<endl<<endl;
            
            // 将平面参数存储到结果中（使用方法1的结果）
            cam_planes.insert(pair<int,vector<Eigen::Vector4f>>(iter->first,tmp_Plane));
            cam_planes_=cam_planes;
        }
    }
}

#endif

