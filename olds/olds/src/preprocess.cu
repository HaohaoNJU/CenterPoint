#include <iostream>
#include <stdio.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <config.h>
#include <preprocess.h>
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))



// FIRST OF ALL , DEFINE  LOCK-RELATED STRUCTURE
struct Lock
{
    int *mutex;
    Lock()
    {
        int state = 0;
        cudaMalloc((void**)&mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int),cudaMemcpyHostToDevice);
    }
    ~Lock()
    {
        cudaFree(mutex);
    }
    __device__ void lock()
    {
        while(atomicCAS(mutex,0,1) !=0);
    }
    __device__ void unlock()
    {
        atomicExch(mutex,0);
    }
};





__global__ void IndiceResetKernel(int* indices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x ;
    if(idx < MAX_PILLARS) 
        indices[idx] = -1;
}

__global__ void Point2BEVIdxKernel (float* points, int* _PBEVIdxs,bool* _PMask, int pointNum )
{
    int point_idx =  threadIdx.x + blockIdx.x * blockDim.x ;

    if (point_idx < pointNum)
    {
    float x = points[point_idx * POINT_DIM + 0];    
    float y = points[point_idx * POINT_DIM + 1];    
    float z = points[point_idx * POINT_DIM + 2];    

    if(x >= X_MIN && x <= X_MAX && y >= Y_MIN && y <= Y_MAX && z >= Z_MIN && z <= Z_MAX) 
    {
        int xIdx = int((x-X_MIN)/X_STEP);
        int yIdx = int((y-Y_MIN)/Y_STEP);
        // get BEVIndex of voxels
        int bevIdx = yIdx*BEV_W+xIdx;
        _PMask[point_idx] = true;
        _PBEVIdxs[point_idx] =  bevIdx;
    }
    }
}


__global__ void BEV2VIdxKernel (int* _VBEVIdxs, int* _VRange,int*  _BEVVoxelIdx)
{
    int idx =  threadIdx.x + blockIdx.x * blockDim.x ;
    if (idx < MAX_PILLARS)
    {
        int bev_idx = _VBEVIdxs[idx] ;
        if (bev_idx >= 0) 
        {
            int voxel_idx = _VRange[idx];
            _BEVVoxelIdx[bev_idx] = voxel_idx+1; // TODO : Note that BEVVoxelIdx save valid values begin from 1 
        }
    }
}

__device__ int ReadAndAdd(int* address, int val)
{
    int old = *address;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed,
                                    val + assumed);
    } while (assumed != old);
    return old;
}

// Note that the below func is not valid 
// __device__ int ReadAndAdd(int* address, int val)
// {
//     int old = *address;
//     int assumed = old;
//     while (assumed == old && assumed < MAX_PIONT_IN_PILLARS);
//     {
//         atomicCAS(address, assumed,
//                                     val + assumed);
//         assumed = *address;
//     } 
//     return old;
// }

__global__ void CountAndSumKernel (float* points, int* _BEVVoxelIdx,  bool* _PMask, int* _PBEVIdxs, int* _PPointNumAssigned, float* _VPointSum, int* _VPointNum, int pointNum)
{
    
    int point_idx =  threadIdx.x + blockIdx.x * blockDim.x ;
    if (point_idx < pointNum && _PMask[point_idx])
    {
        // from xyz to bev idx
        float x = points[point_idx * POINT_DIM + 0];    
        float y = points[point_idx * POINT_DIM + 1];    
        float z = points[point_idx * POINT_DIM + 2];    
        int xIdx = int((x-X_MIN)/X_STEP);
        int yIdx = int((y-Y_MIN)/Y_STEP);
        // get BEVIndex of voxels
        int bev_idx = yIdx*BEV_W+xIdx;
        int voxel_idx = _BEVVoxelIdx[bev_idx]-1; // decode voxel_idx
        
        _PBEVIdxs[point_idx] = bev_idx;
        // use threadfence() to make it sequential between blocks
        int voxel_point_idx = ReadAndAdd(_VPointNum+voxel_idx, 1);
        __threadfence();

        if (voxel_point_idx < MAX_PIONT_IN_PILLARS) {
            _PPointNumAssigned[point_idx] = voxel_point_idx;

            atomicAdd(_VPointSum+voxel_idx*3 + 0, x);
            __threadfence();
            atomicAdd(_VPointSum+voxel_idx*3 + 1, y);
            __threadfence();
            atomicAdd(_VPointSum+voxel_idx*3 + 2, z);
            __threadfence();        
        }

        else
            {
                _VPointNum[voxel_idx] = MAX_PIONT_IN_PILLARS;
                _PMask[point_idx] = false;

            }
    }
}

__global__ void PointAssignKernel(float* points, float* feature,int* _BEVVoxelIdx, bool* _PMask,int* _PBEVIdxs, int*  _PPointNumAssigned, float* _VPointSum, int* _VPointNum,int pointNum)
{
    int point_idx =  threadIdx.x + blockIdx.x * blockDim.x ;
    if (point_idx < pointNum && _PMask[point_idx])
    {
        // from xyz to bev idx
        float x = points[point_idx * POINT_DIM + 0];    
        float y = points[point_idx * POINT_DIM + 1];    
        float z = points[point_idx * POINT_DIM + 2];    
        int bev_idx = _PBEVIdxs[point_idx];
        int voxel_idx = _BEVVoxelIdx[bev_idx] -1;
        int voxel_point_idx = _PPointNumAssigned[point_idx];
        
        int voxel_point_num = _VPointNum[voxel_idx] ;
        voxel_point_num = voxel_point_num > MAX_PIONT_IN_PILLARS ? MAX_PIONT_IN_PILLARS : voxel_point_num;
        // TODO ::: 
        if (voxel_idx>=0) 
        {
    
            feature[        voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = x;
            feature[ 1+  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = y;
            feature[ 2+  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = z;
            feature[ 3+  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = points[point_idx * POINT_DIM + 3];
            feature[ 4+  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = points[point_idx * POINT_DIM + 4];

            feature[ 5+  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = x - _VPointSum[voxel_idx * 3 + 0]/voxel_point_num;
            feature[ 6+  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = y - _VPointSum[voxel_idx * 3 + 1]/voxel_point_num;
            feature[ 7+  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = z - _VPointSum[voxel_idx * 3 + 2]/voxel_point_num;

            int x_idx = bev_idx % BEV_W;
            int y_idx = bev_idx / BEV_W;
            feature[8 +  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = x - (x_idx*X_STEP + X_MIN + X_STEP/2); //  x residual to geometric center
            feature[9 +  voxel_idx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ voxel_point_idx* FEATURE_NUM] = y - (y_idx*Y_STEP + Y_MIN + Y_STEP/2); //  y residual to geometric center
        }
    }
}




// void _preprocess_gpu(float* points, float* feature, int* _VBEVIdxs, int pointNum)
void _preprocess_gpu(float* points, float* feature, int* _VBEVIdxs,
 bool* _PMask, int* _PBEVIdxs, int* _PPointNumAssigned, int* _BEVVoxelIdx, float* _VPointSum, int* _VRange, int* _VPointNum,
int pointNum)
{


    cudaMemset(_PBEVIdxs, 0, pointNum * sizeof(int));
    cudaMemset(_PPointNumAssigned, 0, pointNum * sizeof(int));
    cudaMemset(_PMask, 0, pointNum * sizeof(bool));
    cudaMemset(_BEVVoxelIdx, 0, BEV_H * BEV_W * sizeof(int));

    // cudaMalloc((void**)&_VPointSum, MAX_PILLARS * 3 *sizeof(float));
    cudaMemset(_VPointSum, 0, MAX_PILLARS * 3 * sizeof(float));
    cudaMemset(_VPointNum, 0, MAX_PILLARS *  sizeof(int));

    // cudaMalloc((void**)&_VRange, MAX_PILLARS * sizeof(int));
    // cudaMalloc((void**)&_VPointNum, MAX_PILLARS * sizeof(int));

    // compute the time 


    int threadNum= 1024;
    int blockNum = DIVUP(pointNum,threadNum);
    // init _VBEVIdxs
    // IndiceResetKernel<<<DIVUP(MAX_PILLARS, threadNum), threadNum>>>(_VBEVIdxs);
    cudaMemset(_VBEVIdxs, -1 , MAX_PILLARS * sizeof(int));

    // get _PBEVIDxs, _PMask
    Point2BEVIdxKernel<<<blockNum, threadNum>>>(points,_PBEVIdxs,_PMask, pointNum );

    thrust::sort(thrust::device, _PBEVIdxs, _PBEVIdxs + pointNum, thrust::greater<int>());

    thrust::unique_copy(thrust::device, _PBEVIdxs, _PBEVIdxs + pointNum , _VBEVIdxs);

    thrust::sequence(thrust::device, _VRange, _VRange + MAX_PILLARS);

    // map bev idx to voxel idx 
    BEV2VIdxKernel<<<DIVUP(MAX_PILLARS, threadNum), threadNum>>>(_VBEVIdxs, _VRange, _BEVVoxelIdx);

    // The Key Step 
    CountAndSumKernel<<<blockNum, threadNum>>>(points, _BEVVoxelIdx, _PMask, _PBEVIdxs,_PPointNumAssigned,  _VPointSum, _VPointNum, pointNum);
    PointAssignKernel<<<blockNum, threadNum>>>(points, feature, _BEVVoxelIdx, _PMask,_PBEVIdxs, _PPointNumAssigned,  _VPointSum, _VPointNum, pointNum);


}






















