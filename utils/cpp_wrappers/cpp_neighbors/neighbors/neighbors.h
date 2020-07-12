#include "../../cpp_utils/cloud/cloud.h"
#include "../../cpp_utils/nanoflann/nanoflann.hpp"

#include <set>
#include <cstdint>

using namespace std;

void ordered_neighbors(vector<PointXYZ>& queries,               // 需要求其neighbors的点云
                       vector<PointXYZ>& supports,              // 搜索空间
                       vector<int>& neighbors_indices,          // neighbors点索引
                       float radius);                           // 搜索半径

void batch_ordered_neighbors(vector<PointXYZ>& queries,         // 需要求其neighbors的点云,
                             vector<PointXYZ>& supports,        // 搜索空间
                             vector<int>& q_batches,            // 每个点属于第几个batch
                             vector<int>& s_batches,            // 每个点属于第几个batch
                             vector<int>& neighbors_indices,    // neighbors点索引
                             float radius);                     // 搜索半径

void batch_nanoflann_neighbors(vector<PointXYZ>& queries,       // 需要求其neighbors的点云,
                               vector<PointXYZ>& supports,      // 搜索空间
                               vector<int>& q_batches,          // 每个点属于第几个batch
                               vector<int>& s_batches,          // 每个点属于第几个batch
                               vector<int>& neighbors_indices,  // neighbors点索引
                               float radius);
