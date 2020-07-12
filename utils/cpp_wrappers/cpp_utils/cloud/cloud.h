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
//		Cloud header
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


# pragma once           // 前置处理符号，让所在的文件在一个单独的编译中只被包含一次

#include <vector>       // 顺序容器
#include <unordered_map> // 可充当字典，hash表
#include <map>           // 可充当字典，底层用红黑树实现
#include <algorithm>     // 包含一些模板函数
#include <numeric>       // STL标准中基础性的数值算法，均为模板函数
#include <iostream>      // 输入输出流
#include <iomanip>       // I/O控制
#include <cmath>         // 数学运算

#include <time.h>        // 时间处理


// Point类
class PointXYZ
{
public:

	// 点的 x, y, z坐标
	float x, y, z;


	// 构造函数，创建对象时自动执行，类似于python中的 init函数
	// 不提供初始化值时，默认为0
	PointXYZ() { x = 0; y = 0; z = 0; }
	// 提供初始化值，进行赋值操作
	PointXYZ(float x0, float y0, float z0) { x = x0; y = y0; z = z0; }
	
    // 两个点对应坐标相乘并求和
	float dot(const PointXYZ P) const
	{
		return x * P.x + y * P.y + z * P.z;
	}

    // 求该点的 x, y, z坐标的平方和
	float sq_norm()
	{
		return x*x + y*y + z*z;
	}

    // 叉乘
	PointXYZ cross(const PointXYZ P) const
	{
		return PointXYZ(y*P.z - z*P.y, z*P.x - x*P.z, x*P.y - y*P.x);
	}


	// 重载操作符[]，根据i的值读取x, y, z坐标
	float operator [] (int i) const
	{
		if (i == 0) return x;
		else if (i == 1) return y;
		else return z;
	}

    // 重载操作符+=
	PointXYZ& operator+=(const PointXYZ& P)
	{
		x += P.x;
		y += P.y;
		z += P.z;
		return *this;
	}
    // 重载操作符-=
	PointXYZ& operator-=(const PointXYZ& P)
	{
		x -= P.x;
		y -= P.y;
		z -= P.z;
		return *this;
	}
    // 重载操作符*=
	PointXYZ& operator*=(const float& a)
	{
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}
};


// 点与点之间的操作
// 两点相加
inline PointXYZ operator + (const PointXYZ A, const PointXYZ B)
{
	return PointXYZ(A.x + B.x, A.y + B.y, A.z + B.z);
}
// 两点相减
inline PointXYZ operator - (const PointXYZ A, const PointXYZ B)
{
	return PointXYZ(A.x - B.x, A.y - B.y, A.z - B.z);
}
// 点乘以常数
inline PointXYZ operator * (const PointXYZ P, const float a)
{
	return PointXYZ(P.x * a, P.y * a, P.z * a);
}
// 常数乘以点
inline PointXYZ operator * (const float a, const PointXYZ P)
{
	return PointXYZ(P.x * a, P.y * a, P.z * a);
}
// 输出点的坐标
inline std::ostream& operator << (std::ostream& os, const PointXYZ P)
{
	return os << "[" << P.x << ", " << P.y << ", " << P.z << "]";
}
// 判断两个点是否相等
inline bool operator == (const PointXYZ A, const PointXYZ B)
{
	return A.x == B.x && A.y == B.y && A.z == B.z;
}
// 小于等于XXX的最大整数
inline PointXYZ floor(const PointXYZ P)
{
	return PointXYZ(std::floor(P.x), std::floor(P.y), std::floor(P.z));
}

// 函数声明
PointXYZ max_point(std::vector<PointXYZ> points);
PointXYZ min_point(std::vector<PointXYZ> points);


struct PointCloud
{

    std::vector<PointXYZ> pts;

    // 返回点云中点的数目
    inline size_t kdtree_get_point_cound() const
    {
        return pts.size();
    }

    // 返回点云中第idx个点的第dim维的值
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool ketree_get_bbox(BBOX& /* bb */) const {return false;}

}








