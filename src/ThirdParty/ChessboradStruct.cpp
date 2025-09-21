/*  Copyright 2017 onlyliu(997737609@qq.com).                             */
/*                                                                        */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */

#include "ChessboradStruct.h"
#include <fstream>  
#include <limits>
#include<numeric>

#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */

ChessboradStruct::ChessboradStruct()
{

}

ChessboradStruct::~ChessboradStruct()
{

}

inline float distv(cv::Vec2f a, cv::Vec2f b)
{
	return std::sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]));
}

inline float mean_l(std::vector<float> &resultSet)
{
	double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	double mean = sum / resultSet.size(); //ŸùÖµ  
	return mean;
}

inline float stdev_l(std::vector<float> &resultSet, float &mean)
{
	double accum = 0.0;
	mean = mean_l(resultSet);
	std::for_each(std::begin(resultSet), std::end(resultSet), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	double stdev = sqrt(accum / (resultSet.size() - 1)); //·œ²î 
	return stdev;
}

inline float stdevmean(std::vector<float> &resultSet)
{
	float stdvalue, meanvalue;

	stdvalue = stdev_l(resultSet, meanvalue);

	return stdvalue / meanvalue;
}

int ChessboradStruct::directionalNeighbor(int idx, cv::Vec2f v, cv::Mat chessboard, Corners& corners, int& neighbor_idx, float& min_dist)
{

#if 1
	// list of neighboring elements, which are currently not in use
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;
			}
		}

	int nsize = unused.size();

	for (int i = 0; i < nsize;)
	{
		if (unused[i] < 0)
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0;
			nsize = unused.size();
			continue;
		}
		i++;
	}

	std::vector<float> dist_edge;
	std::vector<float> dist_point;

	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	for (int i = 0; i < unused.size(); i++)
	{
		int ind = unused[i];
		cv::Vec2f diri = cv::Vec2f(corners.p[ind].x, corners.p[ind].y) - idxp;
		float disti = diri[0] * v[0] + diri[1] * v[1];

		cv::Vec2f de = diri - disti*v;
		dist_edge.push_back(distv(de, cv::Vec2f(0, 0)));
		// distances
		dist_point.push_back(disti);
	}
#else
	// list of neighboring elements, which are currently not in use
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;//flag the used idx
			}
		}

	std::vector<float> dist_edge;
	std::vector<float> dist_point;

	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	for (int i = 0; i < corners.p.size(); i++)
	{
		if (unused[i] == -1)
		{
			dist_point.push_back(std::numeric_limits<float>::max());
			dist_edge.push_back(0);
			continue;
		}
		cv::Vec2f diri = cv::Vec2f(corners.p[i].x, corners.p[i].y) - idxp;
		float disti = diri[0] * v[0] + diri[1] * v[1];

		cv::Vec2f de = diri - disti*v;
		dist_edge.push_back(distv(de, cv::Vec2f(0, 0)));
		// distances
		dist_point.push_back(disti);
	}

#endif

	// find best neighbor
	int min_idx = 0;
	min_dist = std::numeric_limits<float>::max();

	//min_dist = dist_point[0] + 5 * dist_edge[0];
	for (int i = 0; i < dist_point.size(); i++)
	{
		if (dist_point[i] > 0)
		{
			float m = dist_point[i] + 5 * dist_edge[i];
			if (m < min_dist)
			{
				min_dist = m;
				min_idx = i;
			}
		}
	}
	neighbor_idx = unused[min_idx];

	return 1;
}


cv::Mat ChessboradStruct::initChessboard(Corners& corners, int idx)
{
	// return if not enough corners
	if (corners.p.size() < 9)
	{
		logd("not enough corners!\n");
		chessboard.release();//return empty!
		return chessboard;
	}
	// init chessboard hypothesis
	chessboard = -1 * cv::Mat::ones(3, 3, CV_32S);
	
	// extract feature index and orientation(central element)
	cv::Vec2f v1 = corners.v1[idx];
	cv::Vec2f v2 = corners.v2[idx];
	chessboard.at<int>(1, 1) = idx;
	std::vector<float> dist1(2), dist2(6);

	// find left / right / top / bottom neighbors
	directionalNeighbor(idx, +1 * v1, chessboard, corners, chessboard.at<int>(1, 2), dist1[0]);
	directionalNeighbor(idx, -1 * v1, chessboard, corners, chessboard.at<int>(1, 0), dist1[1]);
	directionalNeighbor(idx, +1 * v2, chessboard, corners, chessboard.at<int>(2, 1), dist2[0]);
	directionalNeighbor(idx, -1 * v2, chessboard, corners, chessboard.at<int>(0, 1), dist2[1]);

	// find top - left / top - right / bottom - left / bottom - right neighbors
	
	directionalNeighbor(chessboard.at<int>(1, 0), -1 * v2, chessboard, corners, chessboard.at<int>(0, 0), dist2[2]);
	directionalNeighbor(chessboard.at<int>(1, 0), +1 * v2, chessboard, corners, chessboard.at<int>(2, 0), dist2[3]);
	directionalNeighbor(chessboard.at<int>(1, 2), -1 * v2, chessboard, corners, chessboard.at<int>(0, 2), dist2[4]);
	directionalNeighbor(chessboard.at<int>(1, 2), +1 * v2, chessboard, corners, chessboard.at<int>(2, 2), dist2[5]);

	// initialization must be homogenously distributed
		

		bool sigood = false;
		sigood = sigood||(dist1[0]<0) || (dist1[1]<0);
		sigood = sigood || (dist2[0]<0) || (dist2[1]<0) || (dist2[2]<0) || (dist2[3]<0) || (dist2[4]<0) || (dist2[5]<0);
		

		sigood = sigood || (stdevmean(dist1) > 0.3) || (stdevmean(dist2) > 0.3);

		if (sigood == true)
		{
			chessboard.release();
			return chessboard;
		}
		return chessboard;
}

float ChessboradStruct::chessboardEnergy(cv::Mat chessboard, Corners& corners)
{
         float lamda = m_lamda;
	//energy: number of corners
	float E_corners = -1 * chessboard.size().area();
	//energy: structur
	float E_structure = 0;
	//walk through rows
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols-2; j++)
		{
			std::vector<cv::Vec2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(i, k);
				x.push_back(corners.p[n]);
			}
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0,0));
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	//walk through columns
	for (int i = 0; i < chessboard.cols; i++)
		for (int j = 0; j < chessboard.rows-2; j++)
		{
			std::vector<cv::Vec2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(k, i);
				x.push_back(corners.p[n]);
			}
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0, 0));
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	// final energy
	float E = E_corners + lamda*chessboard.size().area()*E_structure;

	return E;
}

// replica prediction(new)
void ChessboradStruct::predictCorners(std::vector<cv::Vec2f>& p1, std::vector<cv::Vec2f>& p2, 
	std::vector<cv::Vec2f>& p3, std::vector<cv::Vec2f>& pred)
{
	cv::Vec2f v1, v2;
	float a1, a2, a3;
	float s1, s2, s3;
	pred.resize(p1.size());
	for (int i = 0; i < p1.size(); i++)
	{
		// compute vectors
		v1 = p2[i] - p1[i];
		v2 = p3[i] - p2[i];
		// predict angles
		a1 = atan2(v1[1], v1[0]);
		a2 = atan2(v1[1], v1[0]);
		a3 = 2.0 * a2 - a1;

		//predict scales
		s1 = distv(v1, cv::Vec2f(0, 0));
		s2 = distv(v2, cv::Vec2f(0, 0));
		s3 = 2 * s2 - s1;
		pred[i] = p3[i] + 0.75*s3*cv::Vec2f(cos(a3), sin(a3));
	}
}

void ChessboradStruct::assignClosestCorners(std::vector<cv::Vec2f>&cand, std::vector<cv::Vec2f>&pred, std::vector<int> &idx)
{
	//return error if not enough candidates are available
	if (cand.size() < pred.size())
	{
		idx.resize(1);
		idx[0] = -1;
		return;
	}
	idx.resize(pred.size());

	//build distance matrix
	cv::Mat D = cv::Mat::zeros(cand.size(), pred.size(), CV_32FC1);
	float mind = FLT_MAX;
	for (int i = 0; i < D.cols; i++)//ÁÐÓÅÏÈ
	{
		cv::Vec2f delta;
		for (int j = 0; j < D.rows; j++)
		{
			delta = cand[j] - pred[i];
			float s = distv(delta, cv::Vec2f(0, 0));
			D.at<float>(j, i) = s;
			if (s < mind)
			{
				mind = s;
			}
		}
	}
	
	// search greedily for closest corners
	for (int k = 0; k < pred.size(); k++)
	{
		bool isbreak = false;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (fabs(D.at<float>(i, j) - mind) < 10e-10)
				{
					idx[j] = i;
					for (int m = 0; m < D.cols; m++)
					{
						D.at<float>(i, m) = FLT_MAX;
					}
					for (int m = 0; m < D.rows; m++)
					{
						D.at<float>(m,j) = FLT_MAX;
					}
					isbreak = true;
					break;
				}
			}
			if (isbreak == true)
				break;
		}
		mind = FLT_MAX;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (D.at<float>(i, j) < mind)
				{
					mind = D.at<float>(i, j);
				}
			}
		}
	}
}



cv::Mat ChessboradStruct::growChessboard(cv::Mat chessboard, Corners& corners, int border_type)
{
	if (chessboard.empty() == true)
	{
		return chessboard;
	}
	std::vector<cv::Point2f> p = corners.p;
	// list of  unused feature elements
	std::vector<int> unused(p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;
			}
		}

	int nsize = unused.size();

	for (int i = 0; i < nsize; )
	{
		if (unused[i] < 0)
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0; 
			nsize = unused.size();
			continue;
		}
		i++;
	}

	// candidates from unused corners
	std::vector<cv::Vec2f> cand;
	for (int i = 0; i < unused.size(); i++)
	{
		cand.push_back(corners.p[unused[i]]);
	}
	// switch border type 1..4
	cv::Mat chesstemp;

	switch (border_type)
	{
	case 0:
	{
		std::vector<cv::Vec2f> p1, p2, p3,pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == chessboard.cols - 3)
				{				
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 0, 1, 0,0);

		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, chesstemp.cols - 1) = unused[idx[i]];//ÓÒ
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 1:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row == chessboard.rows - 3)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 1, 0, 0, 0, 0);
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(chesstemp.rows - 1, i) = unused[idx[i]];//ÏÂ
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 2:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == 2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 1, 0, 0, 0);//×ó
		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, 0) = unused[idx[i]];
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 3:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row ==  2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}
		cv::copyMakeBorder(chessboard, chesstemp, 1, 0, 0, 0, 0, 0);//ÉÏ
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(0, i) = unused[idx[i]];
		}
		chessboard = chesstemp.clone();
		break;
	}
	default:
		break;
	}
	return chessboard;
}




/**
 * @brief 从检测到的角点中构建棋盘格结构
 * 
 * 该函数是棋盘格检测的核心算法，采用基于能量优化的方法从角点集合中识别和构建棋盘格。
 * 算法流程：初始化候选棋盘格 → 能量优化扩展 → 重叠检测与处理 → 最终筛选
 * 
 * @param corners 输入参数，包含所有检测到的角点信息（位置、响应强度等）
 * @param chessboards 输出参数，存储识别出的棋盘格结构，每个Mat表示一个棋盘格
 * @param lamda 能量函数权重参数，控制几何约束的严格程度
 *              - 值越大：几何约束越严格，要求更规整的棋盘格
 *              - 值越小：几何约束越宽松，允许更多变形的棋盘格
 *              - 推荐范围：0.8-2
 */
void ChessboradStruct::chessboardsFromCorners( Corners& corners, std::vector<cv::Mat>& chessboards, float lamda)
{
	logd("Structure recovery:\n");
    m_lamda = lamda;  // 保存能量函数权重参数
    
    // std::cout << "\n=== 棋盘格构建过程调试 ===" << std::endl;
    // std::cout << "输入角点数量: " << corners.p.size() << std::endl;
    // std::cout << "能量权重参数lamda: " << lamda << std::endl;
    
    int valid_initial_count = 0;
    int energy_pass_count = 0;
    int final_quality_count = 0;
    int added_chessboards = 0;
	
	// 第一阶段：遍历所有角点，尝试以每个角点为起点构建棋盘格
	for (int i = 0; i < corners.p.size(); i++)
	{
		// 调试信息：每128个角点输出一次进度
		//if (i % 128 == 0)
		//	printf("%d, %d\n", i, corners.p.size());//fyy
	
		// 步骤1：以当前角点为起点初始化一个候选棋盘格
		// initChessboard会尝试找到该角点的最近邻角点，构建2x2的初始棋盘格
		cv::Mat csbd = initChessboard(corners, i);
		if (csbd.empty() == true)
		{
			continue;  // 如果无法初始化棋盘格，跳过当前角点
		}
		valid_initial_count++;
		
		// 步骤2：计算初始棋盘格的能量值
		// 能量值越小表示棋盘格质量越好，能量值>0表示质量太差
		float E = chessboardEnergy(csbd, corners);
		if (E > 0){ continue; }  // 能量值>0，跳过质量差的候选棋盘格
		energy_pass_count++;
		
		// 步骤3：从初始棋盘格开始，通过能量优化进行扩展
		cv::Mat chessboard = csbd.clone();  // 复制初始棋盘格
		int s = 0;  // 扩展步数计数器
		
		// 第二阶段：迭代扩展棋盘格，直到无法进一步优化
		while (true)
		{
			s++;
			// 计算当前棋盘格的能量值
			float energy = chessboardEnergy(chessboard, corners);

			// 步骤4：生成4个方向的扩展候选方案
			// 分别尝试向上、下、左、右四个方向扩展棋盘格
			std::vector<cv::Mat> proposal(4);
			std::vector<float> p_energy(4);
			
			// 计算每个扩展方案的能量值
			for (int j = 0; j < 4; j++)
			{
				proposal[j] = growChessboard(chessboard, corners, j);  // 生成第j个方向的扩展方案
				p_energy[j] = chessboardEnergy(proposal[j], corners);  // 计算该方案的能量值
			}
			
			// 步骤5：选择能量值最小的扩展方案（最优方案）
			float min_value = p_energy[0];
			int min_idx = 0;
			for (int i0 = 1; i0 < p_energy.size(); i0++)
			{
				if (min_value > p_energy[i0])
				{
					min_value = p_energy[i0];
					min_idx = i0;
				}
			}
			
			// 步骤6：如果最优方案的能量值比当前棋盘格更低，则接受该方案
			cv::Mat chessboardt;
			if (p_energy[min_idx] < energy)
			{
				chessboardt = proposal[min_idx];
				chessboard = chessboardt.clone();  // 更新当前棋盘格
			}
			else
			{
				break;  // 无法进一步优化，退出扩展循环
			}
		}//end while

		// 第三阶段：质量筛选，只保留高质量的棋盘格
		// 能量值<-10表示棋盘格质量足够好，可以加入候选列表
		float final_energy = chessboardEnergy(chessboard, corners);
		if (final_energy < -10)
		{
			final_quality_count++;
			// 第四阶段：重叠检测与处理
			// 检查新发现的棋盘格是否与已存在的棋盘格有重叠
			cv::Mat overlap = cv::Mat::zeros(cv::Size(2, chessboards.size()), CV_32FC1);
			
			// 遍历所有已存在的棋盘格，检查重叠情况
			for (int j = 0; j < chessboards.size(); j++)
			{
				bool isbreak = false;
				// 检查当前棋盘格与第j个已存在棋盘格是否有共同角点
				for (int k = 0; k < chessboards[j].size().area(); k++)
				{
					int refv = chessboards[j].at<int>(k / chessboards[j].cols, k % chessboards[j].cols);
					for (int l = 0; l < chessboard.size().area(); l++)
					{
						int isv = chessboard.at<int>(l / chessboard.cols, l % chessboard.cols);
						if (refv == isv)  // 发现共同角点，表示有重叠
						{
							overlap.at<float>(j, 0) = 1.0;  // 标记有重叠
							float s = chessboardEnergy(chessboards[j], corners);  // 计算已存在棋盘格的能量
							overlap.at<float>(j, 1) = s;  // 保存能量值用于比较
							isbreak = true;
							break;
						}
					}
				}
			}//endfor

			// 第五阶段：根据重叠情况决定是否添加新棋盘格
			// 检查是否存在重叠
			bool isoverlap = false;
			for (int i0 = 0; i0 < overlap.rows; i0++)
			{
				if (overlap.empty() == false)
				{
					if (fabs(overlap.at<float>(i0, 0)) > 0.000001)  // 检查是否有重叠标记
					{
						isoverlap = true;
						break;
					}
				}
			}
			
			if (isoverlap == false)
			{
				// 情况1：无重叠，直接添加新棋盘格
				chessboards.push_back(chessboard);
				added_chessboards++;
			}
			else
			{
				// 情况2：存在重叠，需要比较质量并决定替换策略
				bool flagpush = true;  // 是否应该添加新棋盘格
				std::vector<bool> flagerase(overlap.rows);  // 标记哪些已存在棋盘格应该被删除
				for (int m = 0; m < flagerase.size(); m++)
				{
					flagerase[m] = false;
				}
				
				float ce = chessboardEnergy(chessboard, corners);  // 新棋盘格的能量值
				
				// 比较新棋盘格与重叠的已存在棋盘格的质量
				for (int i1 = 0; i1 < overlap.rows; i1++)
				{
					if (fabs(overlap.at<float>(i1, 0)) > 0.0001)  // 如果存在重叠
					{	
						bool isb1 = overlap.at<float>(i1, 1) > ce;  // 比较能量值

						// 使用整数比较避免浮点数精度问题
						int a = int(overlap.at<float>(i1, 1) * 1000);
						int b = int(ce * 1000);

						bool isb2 = a > b;
						if (isb1 != isb2)
							printf("find bug!\n");  // 调试信息

						if (isb2)  // 如果新棋盘格质量更好（能量值更小）
						{	
							flagerase[i1] = true;  // 标记删除已存在的棋盘格
						}
						else  // 如果已存在棋盘格质量更好
						{
							flagpush = false;  // 不添加新棋盘格
						}
					}
				}//end for

				// 执行替换操作
				if (flagpush == true)  // 如果决定添加新棋盘格
				{
					// 删除质量较差的已存在棋盘格
					for (int i1 = 0; i1 < chessboards.size();)
					{
						std::vector<cv::Mat>::iterator it = chessboards.begin() + i1;
						std::vector<bool>::iterator it1 = flagerase.begin() + i1;
						if (*it1 == true)  // 如果标记为删除
						{
							chessboards.erase(it);  // 删除棋盘格
							flagerase.erase(it1);   // 删除标记
							i1 = 0;  // 重新开始遍历（因为删除了元素）
						}
						i1++;
					}
					chessboards.push_back(chessboard);  // 添加新棋盘格
					added_chessboards++;
				}
			}//endif
		}//endif
	}//end for 
	
	// 输出统计信息
	// std::cout << "\n=== 棋盘格构建统计 ===" << std::endl;
	// std::cout << "遍历角点数量: " << corners.p.size() << std::endl;
	// std::cout << "成功初始化棋盘格数量: " << valid_initial_count << std::endl;
	// std::cout << "通过初始能量检查数量: " << energy_pass_count << std::endl;
	// std::cout << "通过最终质量检查数量: " << final_quality_count << std::endl;
	// std::cout << "最终添加的棋盘格数量: " << added_chessboards << std::endl;
	// std::cout << "当前棋盘格总数: " << chessboards.size() << std::endl;
	// 
	// if(chessboards.size() == 0) {
	// 	std::cout << "\n❌ 没有检测到棋盘格的可能原因:" << std::endl;
	// 	if(valid_initial_count == 0) {
	// 		std::cout << "  - 角点数量不足，无法初始化任何棋盘格" << std::endl;
	// 		std::cout << "  - 建议：降低角点检测阈值" << std::endl;
	// 	} else if(energy_pass_count == 0) {
	// 		std::cout << "  - 初始棋盘格质量都太差（能量值>0）" << std::endl;
	// 		std::cout << "  - 建议：降低棋盘格识别阈值lamda" << std::endl;
	// 	} else if(final_quality_count == 0) {
	// 		std::cout << "  - 扩展后的棋盘格质量仍不达标（能量值>-10）" << std::endl;
	// 		std::cout << "  - 建议：调整能量阈值或改善图像质量" << std::endl;
	// 	} else {
	// 		std::cout << "  - 棋盘格被重叠检测过滤掉了" << std::endl;
	// 		std::cout << "  - 建议：检查重叠检测逻辑或图像中的棋盘格布局" << std::endl;
	// 	}
	// }
}
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
void ChessboradStruct::drawchessboard(cv::Mat img, Corners& corners, std::vector<cv::Mat>& chessboards, char * title, int t_, cv::Rect rect)
{
        printf("end!\n");
        
	cv::RNG rng(0xFFFFFFFF);
	std::string s("If it's useful, please give a star ^-^.");
	std::string s1("https://github.com/onlyliucat\n");
	//std::cout<<BOLDBLUE<<s<<std::endl<<BOLDGREEN<<s1<<std::endl;//fyy
	cv::Mat disp = img.clone();

	if (disp.channels() < 3)
		cv::cvtColor(disp, disp, CV_GRAY2BGR);
	float scale = 0.3;
	int n = 8;
	if (img.rows < 2000 || img.cols < 2000)
	{
		scale = 1;
		n = 2;
	}
	for (int k = 0; k < chessboards.size(); k++)
	{
		cv::Scalar s(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
		s = s * 255;
	
		for (int i = 0; i < chessboards[k].rows; i++)
			for (int j = 0; j < chessboards[k].cols; j++)
			{
				int d = chessboards[k].at<int>(i, j);
				//cv::circle(disp, cv::Point2f(corners.p[d].x + rect.x, corners.p[d].y + rect.y), n, s, n);
				cv::putText(disp,std::to_string(d),cv::Point2f(corners.p[d].x + rect.x, corners.p[d].y + rect.y),cv::FONT_HERSHEY_SIMPLEX,0.3,cv::Scalar(255,255,255),1,1);
			}
	}
	cv::Mat SmallMat;
	cv::resize(disp, SmallMat, cv::Size(), scale, scale);
	cv::namedWindow(title);
	cv::imshow(title, SmallMat);
	cv::waitKey(t_);
}



