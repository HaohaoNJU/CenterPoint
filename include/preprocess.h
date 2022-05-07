#ifndef __CENTERPOINT_PREPROCESS__
#define __CENTERPOINT_PREPROCESS__
#include <iostream>
#include <fstream>
#include <sstream>
#include "config.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
using namespace std;
#define GPU_CHECK(ans)                                                                                                                               \
  {                                                                                                                                                                                 \                                      
    GPUAssert((ans), __FILE__, __LINE__);                                                                                                 \
  }
                                                                                                                                                                                   
inline void GPUAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}


void _preprocess_gpu(float* points, float* feature,int* indices,
 bool* p_mask, int* p_bev_idx, int* p_point_num_assigned, int* bev_voxel_idx, float* v_point_sum, int* v_range, int* v_point_num,
int pointNum);

void preprocessGPU(float* points, float* feature,int* indices,
 bool* p_mask, int* p_bev_idx, int* p_point_num_assigned, int* bev_voxel_idx, float* v_point_sum, int* v_range, int* v_point_num,
int pointNum, int pointDim);


void preprocess(float* points, float* feature, int* indices, int pointNum, int pointDim);

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum, int pointDim );

#endif