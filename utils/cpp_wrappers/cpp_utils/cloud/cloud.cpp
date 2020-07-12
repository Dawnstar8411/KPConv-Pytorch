//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 : 
//			> 
//
//---------------------------------------------------
//
//		Cloud source :
//		Define usefull Functions/Methods
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


#include "cloud.h"


// 求所有点中x,y,z的最大值
// 输入是一个vector,其中元素为一个PointXYZ类型的点
PointXYZ max_point(std::vector<PointXYZ> points)
{
	// 将vector中的第一个点作为初始点
	PointXYZ maxP(points[0]);

	for (auto p : points)
	{
		if (p.x > maxP.x)
			maxP.x = p.x;

		if (p.y > maxP.y)
			maxP.y = p.y;

		if (p.z > maxP.z)
			maxP.z = p.z;
	}

	return maxP;
}

// 求所有点中x,y,z的最大值
// 输入是一个vector,其中元素为一个PointXYZ类型的点
PointXYZ min_point(std::vector<PointXYZ> points)
{
	// 将vector中的第一个点作为初始点
	PointXYZ minP(points[0]);

	for (auto p : points)
	{
		if (p.x < minP.x)
			minP.x = p.x;

		if (p.y < minP.y)
			minP.y = p.y;

		if (p.z < minP.z)
			minP.z = p.z;
	}

	return minP;
}