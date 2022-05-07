#include"preprocess.h"
#include <string>
#include <sys/time.h>
#include <chrono>
#include <thread>
#include <vector>
#include "logger.h"
#include "iostream"
#include <fstream>
#include <iostream>
#include <sstream>
#include "common.h"

 #define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


// feature : voxels,  indices : coords 
void PreprocessWorker(float* points, float* feature, int* indices, int pointNum, int threadIdx, int pillarsPerThread, int pointDim ){
    // 0 ~ MAX_PIONT_IN_PILLARS

    unsigned short pointCount[MAX_PILLARS] = {0};
    // 0 ~ MAX_PILLARS
    int pillarsIndices[BEV_W*BEV_H] = {0};
    for(size_t idx = 0; idx < BEV_W*BEV_H; idx++){
        pillarsIndices[idx] = -1;}

    int pillarCount = threadIdx*pillarsPerThread;
    for(int idx = 0; idx < pointNum; idx++){
        float x = points[idx*pointDim];
        float y = points[idx*pointDim+1];
        float z = points[idx*pointDim+2];

 
        if(pillarCount> MAX_PILLARS - 1)
           continue;

        if(x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || 
           z < Z_MIN || z > Z_MAX)
           continue;

        int xIdx = int((x-X_MIN)/X_STEP);
        int yIdx = int((y-Y_MIN)/Y_STEP);
        
        if(xIdx % THREAD_NUM != threadIdx)
            continue;

        // get Real Index of voxels
        int pillarIdx = yIdx*BEV_W+xIdx;
        // pillarCountIdx default is -1 
        auto pillarCountIdx = pillarsIndices[pillarIdx];

        // pillarCountIdx, actual used pillar index, according pillar orders that has been pushed into points, 
        if(pillarCountIdx == -1){
            pillarCountIdx = pillarCount;
            // indices[pillarCount*2] = pillarIdx;
            pillarsIndices[pillarIdx] = pillarCount;
            indices[pillarCount] = pillarIdx;
            ++pillarCount;
        }


        // pointNumInPillar default is 0
        auto pointNumInPillar = pointCount[pillarCountIdx];

       if(pointNumInPillar > MAX_PIONT_IN_PILLARS - 1)
           continue;


        feature[     pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = x;
        feature[1 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = y;
        feature[2 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = z; // z
        feature[3 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = points[idx*pointDim+3]; // instence
        feature[4 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = points[idx*pointDim+4]; // time_lag
        feature[8 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = x - (xIdx*X_STEP + X_MIN + X_STEP/2); //  x residual to geometric center
        feature[9 +  pillarCountIdx*MAX_PIONT_IN_PILLARS * FEATURE_NUM+ pointNumInPillar* FEATURE_NUM] = y - (yIdx*Y_STEP + Y_MIN + Y_STEP/2); //  y residual to geometric center

        ++pointNumInPillar;
        pointCount[pillarCountIdx] = pointNumInPillar;

    }

    for(int pillarIdx = threadIdx*pillarsPerThread; pillarIdx < (threadIdx+1)*pillarsPerThread; pillarIdx++)
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


void preprocess(float* points, float* feature, int* indices, int pointNum, int pointDim =5)
{

    if (MAX_PILLARS%THREAD_NUM) {
        sample::gLogError<< "THREAD_NUM SHOULD EXACTLY DIVIDE MAX_PILLARS ! " << std::endl;
        return ; 
    }
    int size_ =  BEV_W * BEV_H;
    for(int idx=0; idx< static_cast<int>(MAX_PILLARS); idx++){
        indices[idx] = -1;
    }


    for(int idx=0; idx<static_cast<int>( MAX_PILLARS)*static_cast<int>(FEATURE_NUM)*static_cast<int>(MAX_PIONT_IN_PILLARS); idx++){
        feature[idx] = 0;
    }




    std::vector<std::thread> threadPool;
    for(int idx=0; idx <static_cast<int>(  THREAD_NUM); idx++){
        std::thread worker(PreprocessWorker,
                                             points,
                                             feature,
                                             indices,
                                             pointNum,
                                             idx,
                                             MAX_PILLARS/THREAD_NUM,
                                             pointDim
                                             );
        
        threadPool.push_back(std::move(worker));
    }

    for(auto idx=0; idx < THREAD_NUM; idx++){
        threadPool[idx].join();
    }



}


void preprocessGPU(float* dev_points, float* feature,int* indices,
 bool* p_mask, int* p_bev_idx, int* p_point_num_assigned, int* bev_voxel_idx, float* v_point_sum, int* v_range, int* v_point_num,
int pointNum, int pointDim = 5)
{
    pointNum = pointNum > MAX_POINTS ? MAX_POINTS : pointNum;
    
    GPU_CHECK(cudaMemset(feature, 0.0, MAX_PILLARS * MAX_PIONT_IN_PILLARS * FEATURE_NUM * sizeof(float)));

    _preprocess_gpu( dev_points, feature, indices, 
    p_mask, p_bev_idx,  p_point_num_assigned,  bev_voxel_idx, v_point_sum,  v_range,  v_point_num,
     pointNum);

}


bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum, int pointDim)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        sample::gLogError << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    bufPtr = malloc(fileSize);
    if(bufPtr == nullptr){
        sample::gLogError << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
        return false;
    }
    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    
    pointNum = fileSize /sizeof(float) / pointDim;
    if( fileSize /sizeof(float) % pointDim != 0){
         sample::gLogError << "[Error] File Size Error! " << fileSize << std::endl;
    }
    sample::gLogInfo << "[INFO] pointNum : " << pointNum << std::endl;
    return true;
}
