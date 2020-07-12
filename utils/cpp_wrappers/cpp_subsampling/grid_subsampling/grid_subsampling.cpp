
#include "grid_subsampling.h"


void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,    // 多个点的features顺序排列
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose) {

	// 初始化变量

	size_t N = original_points.size();          // 原始点云中有多少个点
	size_t fdim = original_features.size() / N; // 每个点的feature的长度
	size_t ldim = original_classes.size() / N;  // 每个点的label的长度

	PointXYZ minCorner = min_point(original_points);  // 原始点云中最小的点坐标，是一个虚拟的点
	PointXYZ maxCorner = max_point(original_points);  // 原始点云中最大的点坐标，是一个虚拟的点
	PointXYZ originCorner = floor(minCorner * (1/sampleDl)) * sampleDl; // 将做小的点的坐标标准化到格子的顶点处

	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1; // x轴上划分多少个格子
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1; // y轴上划分多少个格子
	//size_t sampleNZ = (size_t)floor((maxCorner.z - originCorner.z) / sampleDl) + 1; // z轴上划分多少个格子

	bool use_feature = original_features.size() > 0;  // 是否需要同时处理features
	bool use_classes = original_classes.size() > 0;   // 是否需要同时处理labels


	// Create the sampled map

	int i = 0;            // 记录当前点是第几个点
	int nDisp = N / 100;  // 控制打印的变量

	size_t iX, iY, iZ, mapIdx;                  // 初始化变量，点所属的网格的索引
	unordered_map<size_t, SampledData> data;    // 初始化map类型变量data,其中key值为3D网格的索引，SampledData聚合了该网格中的所有点


    // 依次处理输入点云中的每个点
	for (auto& p : original_points)
	{
		iX = (size_t)floor((p.x - originCorner.x) / sampleDl); // 该点属于x轴上的第几个网格
		iY = (size_t)floor((p.y - originCorner.y) / sampleDl); // 该点属于y轴上的第几个网格
		iZ = (size_t)floor((p.z - originCorner.z) / sampleDl); // 该点属于z轴上的第几个网格
		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;      // 该点属于所有网格里的第几个网格

		// 如果字典中没有此元素，则创建一个
		if (data.count(mapIdx) < 1)
			data.emplace(mapIdx, SampledData(fdim, ldim));

        // 将当前点与其features,labels划归到所属的网格中去
		if (use_feature && use_classes)
			data[mapIdx].update_all(p, original_features.begin() + i * fdim, original_classes.begin() + i * ldim);
		else if (use_feature)
			data[mapIdx].update_features(p, original_features.begin() + i * fdim);
		else if (use_classes)
			data[mapIdx].update_classes(p, original_classes.begin() + i * ldim);
		else
			data[mapIdx].update_points(p);

		// 打印处理进程
		i++;
		if (verbose > 1 && i%nDisp == 0)
			std::cout << "\rSampled Map : " << std::setw(3) << i / nDisp << "%";
	}

	// reserve(n): 预分配n个元素的存储空间
	subsampled_points.reserve(data.size());               // 返回下采样的点坐标
	if (use_feature)
		subsampled_features.reserve(data.size() * fdim);  // 返回下采样的点特征
	if (use_classes)
		subsampled_classes.reserve(data.size() * ldim);   // 返回下采样的点类别
	// 依次处理每个网格中的数据，每个网格中生成一个采样点
	for (auto& v : data)
	{
		subsampled_points.push_back(v.second.point * (1.0 / v.second.count));  // 所有点的坐标取平均值
		if (use_feature)  // 对网格中的所有点的特征取平均值
		{
		    float count = (float)v.second.count;
		    transform(v.second.features.begin(),
                      v.second.features.end(),
                      v.second.features.begin(),
                      [count](float f) { return f / count;});
            subsampled_features.insert(subsampled_features.end(),v.second.features.begin(),v.second.features.end());
		}
		if (use_classes)  // 少数服从多数原则，确定采样点的label
		{
		    for (int i = 0; i < ldim; i++)
		        subsampled_classes.push_back(max_element(v.second.labels[i].begin(), v.second.labels[i].end(),
		        [](const pair<int, int>&a, const pair<int, int>&b){return a.second < b.second;})->first);
		}
	}

	return;
}


void batch_grid_subsampling(vector<PointXYZ>& original_points,
                              vector<PointXYZ>& subsampled_points,
                              vector<float>& original_features,
                              vector<float>& subsampled_features,
                              vector<int>& original_classes,
                              vector<int>& subsampled_classes,
                              vector<int>& original_batches,
                              vector<int>& subsampled_batches,
                              float sampleDl,
                              int max_p)
{
    int b = 0
    int sum_b = 0

    size_t N = original_points.size();          // 原始点云中有多少个点
	size_t fdim = original_features.size() / N; // 每个点的feature的长度
	size_t ldim = original_classes.size() / N;  // 每个点的label的长度

    // 处理max_p等于0时的情况
    if (max_p < 1)
       max_p = N;

    // 依次处理一个batch中的点云
    // original_batches中第i个元素为该batch中第i个点云的点的数量
    for (b==0; b < original_batches.size(); b++)
    {
        //取出该batch中第b个点云的数据
        vector<PointXYZ> b_o_points = vector<PointXYZ>(original_points.begin() + sum_b,
                                                       original_points.begin + sum_b + original_batches[b]);
        // 该点云的特征向量
        vector<float> b_o_features;
        if (original_features.size() > 0)
        {
            b_o_features = vector<float>(original_features.begin() + sum_b * fdim,
                                         original_features.begin() + (sum_b + original_batches[b]) * fdim);
        }

        // 该点云的label
        vector<int> b_o_classes;
        if (original_classes.size() > 0)
        {
           b_o_classes = vector<int>(original_classes.begin() + sum_b * ldim,
                                     original_classes.begin() + sum_b + original_batches[b] * ldim);
        }

        vector<PointXYZ> b_s_points;  // 输出的点云
        vector<float> b_s_features;   // 输出的features
        vector<int> b_s_classes;      // 输出的labels

        // 对当前点云进行下采样处理
        grid_subsampling(b_o_points,
                         b_s_points,
                         b_o_features,
                         b_s_features,
                         b_o_classes,
                         b_s_classes,
                         sampleDl,
                         0);

        // 将处理玩的点云重新组合成一个batch的数据
        if ( b_s_points.size() <= max_p)
        {
            subsampled_points.insert(subsampled_points.end(), b_s_points.begin(), b_s_points.end());

            if (original_features.size() > 0)
                subsampled_features.insert(subsampled_features.end(),b_s_features.begin(), b_s_features.end());

            if (original_classes.size() > 0)
                subsampled_classes.insert(subsampled_classes.end(),b_s_classes.begin(), b_s_classes.end());

            subsampled_batches.push_back(b_s_points.size());
        }
        else
        {
            subsampled_points.insert(subsampled_points.end(), b_s_points.begin(), b_s_points.begin() + max_p);

            if (original_features.size() > 0)
                subsampled_features.insert(subsampled_features.end(),b_s_features.begin(), b_s_features.begin()+max_p * fdim);

            if (original_classes.size() > 0)
                subsampled_classes.insert(subsampled_classes.end(),b_s_classes.begin(), b_s_classes.begin() + max_p * ldim);

            subsampled_batches.push_back(max_p);
        }

        // 累加已经处理的点的数目
        sum_b += original_batches[b]
    }

    return;
}
