#ifndef __CENTERPOINT_PREPROCESS__
#define __CENTERPOINT_PREPROCESS__
#include <iostream>
#include <fstream>
#include "config.h"
#include <cuda_runtime_api.h>


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

void PreprocessGPU(float* points, float* feature, int* indices, int pointNum, int pillarNum, int featureNum );

void preprocess(float* points, float* feature, int* indices, int pointNum, int featureNum);

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum, int featureNum );
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