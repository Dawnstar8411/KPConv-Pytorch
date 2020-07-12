#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:


	int count;    // 计数
	PointXYZ point; // 点
	vector<float> features; // 点的特征
	vector<unordered_map<int, int>> labels;  // vector中的每个元素是一个字典，表示一个label

    // 构造函数，无初始化参数，采用默认值
	SampledData() 
	{ 
		count = 0; 
		point = PointXYZ();
	}
    // 构造函数，指定features维度和label的维度
	SampledData(const size_t fdim, const size_t ldim)
	{
		count = 0;
		point = PointXYZ();
	    features = vector<float>(fdim);
	    labels = vector<unordered_map<int, int>>(ldim);
	}

	// 更新点，特征和label
	void update_all(const PointXYZ p, vector<float>::iterator f_begin, vector<int>::iterator l_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	// 更新点和特征
	void update_features(const PointXYZ p, vector<float>::iterator f_begin)
	{
		count += 1; // 数目加1
		point += p; // 累加所有点的坐标
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>()); // 累加特征
		return;
	}
	// 更新点坐标与label，l_begin是存储原始点云labels的vector中对应于当前点的指针
	void update_classes(const PointXYZ p, vector<int>::iterator l_begin)
	{
		count += 1;  // 数目加1
		point += p;  // 累加所有点的坐标
		int i = 0;
		// it是指针，*it表示添加的点的label
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1; // 第*it类的数目加1，后续会根据少数服从多数的原则确定下采样之后点的label
		    i++;
		}
		return;
	}
	// 只更新点坐标
	void update_points(const PointXYZ p)
	{
		count += 1;  // 数量加1
		point += p;  // 累加所有点的坐标
		return;
	}
};



void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose);

void batch_grid_subsampling(vector<PointXYZ>& original_points,
                            vector<PointXYZ>& subsampled_points,
                            vector<float>& original_features,
                            vector<float)& subsampled_features,
                            vector<int>& original_classes,
                            vector<int>& subsampled_classes,
                            vector<int>& original_batches,
                            vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p);