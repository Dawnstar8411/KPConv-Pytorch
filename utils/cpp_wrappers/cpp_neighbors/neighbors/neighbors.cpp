#include "neighbors.h"

void ordered_neighbors(vector<PointXYZ>& queries,
                        vector<PointXYZ>& supports,
                        vector<int>& neighbors_indices,
                        float radius)
{
	float r2 = radius * radius;   // 半径的平方

	int i0 = 0;  // query点索引

	int max_count = 0; // 记录query所能找到的邻居点的最多数目
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());


    // 依次处理每个query点
	for (auto& p0 : queries)
	{
		int i = 0;
		// 遍历整个搜索空间
		for (auto& p : supports)
		{
		    d2 = (p0 - p).sq_norm();  // 计算中心点P0与邻居点p之间的坐标差的平方和
			if (d2 < r2)   // 如果此邻居点在检索半径内
			{
			    // upper_bound利用二分查找法，寻找第一个大于d2的数的位置it
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    // 返回 begin 到 it 之间有多少个元素
			    int index = std::distance(dists[i0].begin(), it);

			    // dists[i0] 是一个vector,其保存了第i0个点的周围邻居点算出来的d2
			    // 在it位置插入，保证vector中的数一直是从小到大排列的
                dists[i0].insert(it, d2);
                // 把p点的索引插入到tmp[i0]合适的位置，使其与dists[i0]中的数相对应
                tmp[i0].insert(tmp[i0].begin() + index, i);

			    // 每个点找到的符合条件的邻居点的个数是不同的，这里持续的记录最大值
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;  // 下一个query点的索引
	}

	// 根据max_count值重新分配空间
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	// 一次处理每一个query点的邻居点
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j]; // 将邻居点的索引依次填入保存返回值的vector中
			else
				neighbors_indices[i0 * max_count + j] = -1;   // 不够max_count的位置用-1来填充
		}
		i0++;  // 下一个query点
	}

	return;
}

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{

	float r2 = radius * radius; // 半径的平方

	int i0 = 0; // query点索引


	int max_count = 0;
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());

	int b = 0;       // 点云在batch中的索引
	int sum_qb = 0;  // query点云分界点
	int sum_sb = 0;  // 搜索空间点云分界点


    // 依次处理每个query点
	for (auto& p0 : queries)
	{
	    // 判断是不是处理完了一个点云，进入了下一个点云
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];  // 更新query点云分界点
	        sum_sb += s_batches[b];  // 更新搜索空间点云分界点
	        b++;
	    }

	    // 只在对应的搜索空间点云中搜索邻居点
	    vector<PointXYZ>::iterator p_it;
		int i = 0;
        for(p_it = supports.begin() + sum_sb; p_it < supports.begin() + sum_sb + s_batches[b]; p_it++ )
        {
		    d2 = (p0 - *p_it).sq_norm();
			if (d2 < r2)
			{
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    int index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, sum_sb + i);

			    // Update max count
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	return;
}

// 利用kdTree进行搜索
void batch_nanoflann_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{

	int i0 = 0;

	float r2 = radius * radius;

	int max_count = 0;
	float d2;
	vector<vector<pair<size_t, float>>> all_inds_dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud current_cloud;

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree_t;

    // Pointer to trees
    my_kd_tree_t* index;

    // Build KDTree for the first batch element
    current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
    index = new my_kd_tree_t(3, current_cloud, tree_params);
    index->buildIndex();


	// Search neigbors indices
	// ***********************

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = true;

	for (auto& p0 : queries)
	{

	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;

	        // Change the points
	        current_cloud.pts.clear();
            current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);

	        // Build KDTree of the current element of the batch
            delete index;
            index = new my_kd_tree_t(3, current_cloud, tree_params);
            index->buildIndex();
	    }

	    // Initial guess of neighbors size
        all_inds_dists[i0].reserve(max_count);

	    // Find neighbors
	    float query_pt[3] = { p0.x, p0.y, p0.z};
		size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);

        // Update max count
        if (nMatches > max_count)
            max_count = nMatches;

        // Increment query idx
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (auto& inds_dists : all_inds_dists)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

		for (int j = 0; j < max_count; j++)
		{
			if (j < inds_dists.size())
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	delete index;

	return;
}



// 返回的邻居点没有按照距离query点从小到大的顺序排列
void brute_neighbors(vector<PointXYZ>& queries, vector<PointXYZ>& supports, vector<int>& neighbors_indices, float radius, int verbose)
{

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	vector<vector<int>> tmp(queries.size());

	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
		int i = 0;
		for (auto& p : supports)
		{
			if ((p0 - p).sq_norm() < r2)
			{
				tmp[i0].push_back(i);
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = -1;
		}
		i0++;
	}

	return;
}

