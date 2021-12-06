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

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))


// __global__ void voxel_assign_kernel(float* points, float* feature, int* indices, // pointNum should smaller than MAX_POINTS
//                int* pillarCount, int* pillarsIndices,              // pillarIndices need set to -1 (H * W)
               
//                )
// {
//     int point_idx = blockIdx.x;
//     int point_num = blockDim.x;
//     float x = points[point_idx * POINT_DIM + 0];    
//     float y = points[point_idx * POINT_DIM + 1];    
//     float z = points[point_idx * POINT_DIM + 2];    
//     if(x >= X_MIN && x <= X_MAX && y >= Y_MIN && y <= Y_MAX && z >= Z_MIN && z <= Z_MAX) 
//     {

//         int xIdx = int((x-X_MIN)/X_STEP);
//         int yIdx = int((y-Y_MIN)/Y_STEP);

//         // get Real Index of voxels
//         int bevIdx = yIdx*BEV_W+xIdx;
//         // pillarCountIdx default is -1 
//         int pillarCountIdx = pillarsIndices[bevIdx];

//         // pillarCountIdx, actual used pillar index, according pillar orders that has been pushed into points, 
//         if(pillarCountIdx == -1){
//             pillarCountIdx = pillarCount[0];
//             pillarsIndices[bevIdx] = pillarCountIdx;
//             indices[pillarCount[0]] = bevIdx;
//             atomicAdd(pillarCount,1);
//         }

//         // pointNumInPillar default is 0
//         auto pointNumInPillar = pointCount[pillarCountIdx];

//        if(pointNumInPillar > MAX_PIONT_IN_PILLARS - 1)
//            continue;

//         feature[     pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = x;
//         feature[1 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = y;
//         feature[2 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = z; // z
//         feature[3 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = points[idx*featureNum+3]; // instence
//         feature[4 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = points[idx*featureNum+4]; // time_lag
//         feature[8 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = x - (xIdx*X_STEP + X_MIN + X_STEP/2); //  x residual to geometric center
//         feature[9 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = y - (yIdx*Y_STEP + Y_MIN + Y_STEP/2); //  y residual to geometric center

//         ++pointNumInPillar;
//         pointCount[pillarCountIdx] = pointNumInPillar;
//     }
// }

__global__ void preprocess_kernel(float* dev_points, float* feature, int* indices, const int pointNum, const int pillarNum, const int featureNum,
int &pillarCount, unsigned short* pointCount, int* pillarsIndices
 ) 
{
    for(int idx = 0; idx < pointNum; idx++){
        float x = dev_points[idx*featureNum];
        float y = dev_points[idx*featureNum+1];
        float z = dev_points[idx*featureNum+2];
        
        if(pillarCount> MAX_PILLARS - 1)
           continue;

        if(x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || 
           z < Z_MIN || z > Z_MAX)
           continue;

        int xIdx = int((x-X_MIN)/X_STEP);
        int yIdx = int((y-Y_MIN)/Y_STEP);
        


        // get Real Index of voxels
        int bevIdx = yIdx*BEV_W+xIdx;
        // pillarCountIdx : from bevIdx to pillarIdx, default is -1 
        auto pillarCountIdx = pillarsIndices[bevIdx];

        // pillarCountIdx, actual used pillar index, according pillar orders that has been pushed into points, 
        if(pillarCountIdx == -1){
            pillarCountIdx = pillarCount;
            // indices[pillarCount*2] = bevIdx;
            pillarsIndices[bevIdx] = pillarCount;
            indices[pillarCount] = bevIdx;
            ++pillarCount;
        }


        // pointNumInPillar default is 0
        auto pointNumInPillar = pointCount[pillarCountIdx];

       if(pointNumInPillar > MAX_PIONT_IN_PILLARS - 1)
           continue;


        feature[     pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = x;
        feature[1 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = y;
        feature[2 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = z; // z
        feature[3 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = dev_points[idx*featureNum+3]; // instence
        feature[4 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = dev_points[idx*featureNum+4]; // time_lag
        feature[8 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = x - (xIdx*X_STEP + X_MIN + X_STEP/2); //  x residual to geometric center
        feature[9 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = y - (yIdx*Y_STEP + Y_MIN + Y_STEP/2); //  y residual to geometric center

        ++pointNumInPillar;
        pointCount[pillarCountIdx] = pointNumInPillar;
    }
    for(int pillarIdx = 0; pillarIdx < pillarNum; pillarIdx++)
    {
        float xCenter = 0;
        float yCenter = 0;
        float zCenter = 0;
        auto pointNum = pointCount[pillarIdx];
        for(int pointIdx=0; pointIdx < pointNum; pointIdx++)
        {
            
            auto x = feature[       pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM];
            auto y = feature[1 + pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM];
            auto z = feature[2 + pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM];
            xCenter += x;
            yCenter += y;
            zCenter += z;
        }

        if (pointNum > 0) {
        xCenter = xCenter / pointNum;
        yCenter = yCenter / pointNum;
        zCenter = zCenter / pointNum;
        }

        
        for(int pointIdx=0; pointIdx < pointNum; pointIdx++)
        {    

            auto x = feature[       pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM];
            auto y = feature[1 + pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM];
            auto z = feature[2 + pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM];


            feature[5 + pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM] = x - xCenter; // x offest from cluster center 
            feature[6 + pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM] = y - yCenter; // y offset ...
            feature[7 + pillarIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointIdx* FEATURE_NUM] = z - zCenter; // z offset ...
        }
    }

}

void PreprocessGPU(float* points, float* feature, int* indices, int pointNum, int pillarNum, int featureNum )
{
    cudaMemset(indices, -1, pillarNum * sizeof(int));
    // points in device
    float* dev_points;
    cudaMalloc((void**)& dev_points, pointNum * sizeof(float));
    cudaMemcpy(dev_points, points, pointNum * sizeof(float),cudaMemcpyHostToDevice);

    // 0 ~ MAX_PIONT_IN_PILLARS
    unsigned short *pointCount; //[MAX_PILLARS] = {0};
    cudaMalloc((void**)&pointCount, MAX_PILLARS * sizeof(unsigned short));
    cudaMemset(pointCount, 0 , MAX_PILLARS * sizeof(unsigned short));

    // 0 ~ MAX_PILLARS
    int *pillarsIndices;//[BEV_W*BEV_H] = {0};
    cudaMalloc((void**)&pillarsIndices, BEV_W*BEV_H * sizeof(int));
    cudaMemset(pillarsIndices, -1, BEV_W*BEV_H * sizeof(int));
    // for(size_t idx = 0; idx < BEV_W*BEV_H; idx++){
    //     pillarsIndices[idx] = -1;}

    int pillarCount = 0; 
    preprocess_kernel<<<1,1>>>(
    dev_points, feature, indices, pointNum, pillarNum, featureNum,
     pillarCount , pointCount, pillarsIndices
    );


    cudaFree(pointCount);
    cudaFree(pillarsIndices);

}





















