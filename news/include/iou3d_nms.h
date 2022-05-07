#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int nms_gpu(const float* boxes_data, long* keep_data, int boxs_num,  float nms_overlap_thresh);
int raw_nms_gpu(const float* reg,  const float* height, const float* dim , const float* rot, const int* indexs, long* keep_data, int boxes_num,  float nms_overlap_thresh);
