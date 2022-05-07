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





// void _preprocess_gpu(float* points, float* feature, int* indices, int pointNum);
// void preprocessGPU(float* points, float* feature, int* indices, int pointNum, int pointDim);
void _preprocess_gpu(float* points, float* feature,int* indices,
 bool* _PMask, int* _PBEVIdxs, int* _PPointNumAssigned, int* _BEVVoxelIdx, float* _VPointSum, int* _VRange, int* _VPointNum,
int pointNum);

void preprocessGPU(float* points, float* feature,int* indices,
 bool* _PMask, int* _PBEVIdxs, int* _PPointNumAssigned, int* _BEVVoxelIdx, float* _VPointSum, int* _VRange, int* _VPointNum,
int pointNum, int pointDim);


void preprocess(float* points, float* feature, int* indices, int pointNum, int pointDim);

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum, int pointDim );
bool readTmpFile(std::string& filename, void*& bufPtr);

template <typename T>
bool saveBinFile(std::string savePath, T* output, size_t shape)
{
    //Save one out node
    std::fstream file(savePath, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cout << "Error opening file." << savePath << std::endl;;
        return false;
    }
    file.write(reinterpret_cast<char *>(output), shape*sizeof(T));
    file.close();
    return true;
}
#endif