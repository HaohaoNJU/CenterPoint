#include "preprocess.h"
#include "postprocess.h"
#include "centerpoint.h"
#include <string>
#include <sys/time.h>
#include <chrono>
#include <thread>
#include <vector>
#include <math.h>
#include "buffers.h"
#include "common.h"
#include <iou3d_nms.h>
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

inline void RotateAroundCenter(Box& box, float (&corner)[4][2], float& cosVal, float& sinVal, float (&cornerANew)[4][2]){
    
    for(auto idx = 0; idx < 4; idx++){
        auto x = corner[idx][0];
        auto y = corner[idx][1];

        cornerANew[idx][0] = (x - box.x) * cosVal + (y - box.y) * (-sinVal) + box.x;
        cornerANew[idx][1] = (x - box.x) * sinVal + (y - box.y) * cosVal + box.y;
    }
}
inline void FindMaxMin(float (&box)[4][2], float& maxVAl, float& minVAl, int xyIdx){
    
    maxVAl = box[0][xyIdx];
    minVAl = box[0][xyIdx];
    
    for(auto idx=0; idx < 4; idx++){
        if (maxVAl < box[idx][xyIdx])
            maxVAl = box[idx][xyIdx];

        if (minVAl > box[idx][xyIdx])
            minVAl = box[idx][xyIdx];
    }
}

inline void AlignBox(float (&cornerRot)[4][2], float (&cornerAlign)[2][2]){

    float maxX = 0;
    float minX = 0;
    float maxY = 0;
    float minY = 0;

    FindMaxMin(cornerRot, maxX, minX, 0); // 0 mean X
    FindMaxMin(cornerRot, maxY, minY, 1); // 1 mean X

    cornerAlign[0][0] = minX;
    cornerAlign[0][1] = minY;
    cornerAlign[1][0] = maxX;
    cornerAlign[1][1] = maxY;
}

inline float IoUBev(Box& boxA, Box& boxB){
   
    float ax1 = boxA.x - boxA.l/2;
    float ax2 = boxA.x + boxA.l/2;
    float ay1 = boxA.y - boxA.w/2;
    float ay2 = boxA.y + boxA.w/2;

    float bx1 = boxB.x - boxB.l/2;
    float bx2 = boxB.x + boxB.l/2;
    float by1 = boxB.y - boxB.w/2;
    float by2 = boxB.y + boxB.w/2;

    float cornerA[4][2] = {{ax1, ay1}, {ax1, ay2},
                         {ax2, ay1}, {ax2, ay2}};
    float cornerB[4][2] = {{bx1, ay1}, {bx1, by2},
                         {bx2, by1}, {bx2, by2}};
    
    float cornerARot[4][2] = {0};
    float cornerBRot[4][2] = {0};

    float cosA = cos(boxA.theta), sinA = sin(boxA.theta);
    float cosB = cos(boxB.theta), sinB = sin(boxB.theta);

    RotateAroundCenter(boxA, cornerA, cosA, sinA, cornerARot);
    RotateAroundCenter(boxB, cornerB, cosB, sinB, cornerBRot);

    float cornerAlignA[2][2] = {0};
    float cornerAlignB[2][2] = {0};

    AlignBox(cornerARot, cornerAlignA);
    AlignBox(cornerBRot, cornerAlignB);
    
    float sBoxA = (cornerAlignA[1][0] - cornerAlignA[0][0]) * (cornerAlignA[1][1] - cornerAlignA[0][1]);
    float sBoxB = (cornerAlignB[1][0] - cornerAlignB[0][0]) * (cornerAlignB[1][1] - cornerAlignB[0][1]);
    
    float interW = std::min(cornerAlignA[1][0], cornerAlignB[1][0]) - std::max(cornerAlignA[0][0], cornerAlignB[0][0]);
    float interH = std::min(cornerAlignA[1][1], cornerAlignB[1][1]) - std::max(cornerAlignA[0][1], cornerAlignB[0][1]);
    
    float sInter = std::max(interW, 0.0f) * std::max(interH, 0.0f);
    float sUnion = sBoxA + sBoxB - sInter;
    
    return sInter/sUnion;
}

void AlignedNMSBev(std::vector<Box>& predBoxs){
    
    if(predBoxs.size() == 0)
        return;

    std::sort(predBoxs.begin(),predBoxs.end(),[ ](Box& box1, Box& box2){return box1.score > box2.score;});

    auto boxSize = predBoxs.size() > INPUT_NMS_MAX_SIZE? INPUT_NMS_MAX_SIZE : predBoxs.size();
    int numBoxValid = 0;
    for(auto boxIdx1 =0; boxIdx1 < boxSize; boxIdx1++){
        
        if (numBoxValid >= OUTPUT_NMS_MAX_SIZE) {
            for(auto boxIdx2 = boxIdx1+1; boxIdx2 < boxSize; boxIdx2++) 
                predBoxs[boxIdx2].isDrop = true;
            break;
        }
        if (predBoxs[boxIdx1].isDrop) continue;


        if (predBoxs[boxIdx1].x >X_CENTER_MAX || predBoxs[boxIdx1].x < X_CENTER_MIN ) {
            predBoxs[boxIdx1].isDrop = true;
            continue;
        } 
        if (predBoxs[boxIdx1].y >Y_CENTER_MAX || predBoxs[boxIdx1].y < Y_CENTER_MIN ) {
            predBoxs[boxIdx1].isDrop = true;
            continue;
        } 

        if (predBoxs[boxIdx1].z >Z_CENTER_MAX || predBoxs[boxIdx1].z < Z_CENTER_MIN ) {
            predBoxs[boxIdx1].isDrop = true;
            continue;
        } 

        for(auto boxIdx2 = boxIdx1+1; boxIdx2 < boxSize; boxIdx2++){
            if(predBoxs[boxIdx2].isDrop == true)
                continue;
            if(IoUBev(predBoxs[boxIdx1], predBoxs[boxIdx2]) > NMS_THREAHOLD)
                predBoxs[boxIdx2].isDrop = true;
        } 
        if (!predBoxs[boxIdx1].isDrop) numBoxValid ++;
    }
}

void postprocess(const samplesCommon::BufferManager& buffers, std::vector<Box>& predResult){


// #define REG_CHANNEL 2
// #define HEIGHT_CHANNEL 1
// #define ROT_CHANNEL 2
// // #define VEL_CHANNEL 2 //don't defined in waymo
// #define DIM_CHANNEL 3

    std::vector<std::string> regName{   "246"};
    std::vector<std::string> rotName{   "258"};

    std::vector<std::string> heightName{"250"};
    std::vector<std::string> dimName{   "264"};

    std::vector<std::string> scoreName{ "265"};
    std::vector<std::string> clsName{   "266"};
for (int taskIdx=0;taskIdx < TASK_NUM;taskIdx++){
        std::vector<Box> predBoxs;
        float* reg = static_cast<float*>(buffers.getHostBuffer(regName[taskIdx]));
        float* height = static_cast<float*>(buffers.getHostBuffer(heightName[taskIdx]));
        float* rot = static_cast<float*>(buffers.getHostBuffer(rotName[taskIdx]));
        // float* vel = static_cast<float*>(buffers.getHostBuffer(velName[taskIdx]));
        float* dim = static_cast<float*>(buffers.getHostBuffer(dimName[taskIdx]));
        float* score = static_cast<float*>(buffers.getHostBuffer(scoreName[taskIdx]));
        int32_t* cls = static_cast<int32_t*>(buffers.getHostBuffer(clsName[taskIdx]));

        int cnt = 0;
        for(size_t yIdx=0; yIdx < OUTPUT_H; yIdx++){
            for(size_t xIdx=0; xIdx < OUTPUT_W; xIdx++){
                auto idx = yIdx* OUTPUT_W + xIdx;
                if(score[idx] < SCORE_THREAHOLD)
                    continue;
                
                float x = (xIdx + reg[0*OUTPUT_H*OUTPUT_W + idx])*OUT_SIZE_FACTOR*X_STEP + X_MIN;
                float y = (yIdx + reg[1*OUTPUT_H*OUTPUT_W + idx])*OUT_SIZE_FACTOR*Y_STEP + Y_MIN;
                float z = height[idx];

                if(x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || z < Z_MIN || z > Z_MAX)
                    continue;
                cnt ++;
                Box box;
                box.x = x;
                box.y = y;
                box.z = z;
                box.l = dim[0*OUTPUT_H*OUTPUT_W + idx];
                box.h = dim[1*OUTPUT_H*OUTPUT_W + idx];
                box.w = dim[2*OUTPUT_H*OUTPUT_W + idx];
                box.theta = atan2(rot[0*OUTPUT_H*OUTPUT_W + idx], rot[1*OUTPUT_H*OUTPUT_W + idx]);

                // box.velX = vel[0*OUTPUT_H*OUTPUT_W+idx];
                // box.velY = vel[1*OUTPUT_H*OUTPUT_W+idx];
                // box.theta = box.theta - PI /2;

                box.score = score[idx];
                box.cls = cls[idx] ; 
                box.isDrop = false;
                predBoxs.push_back(box);
            }
        }
        std::cout << " Num boxes before nms " << cnt << "\n";

        AlignedNMSBev(predBoxs);
        for(auto idx =0; idx < predBoxs.size(); idx++){
            if(!predBoxs[idx].isDrop)
                predResult.push_back(predBoxs[idx]);
        }
        std::cout << " Num boxes after nms " << predResult.size() << "\n";
    }
}




void postprocessGPU(const samplesCommon::BufferManager& buffers,
                                                 std::vector<Box>& predResult ,
                                                 std::map<std::string, std::vector<string>>rpnOutputTensorNames,
                                                 int* dev_score_indexs,
                                                 unsigned long long* mask_cpu,
                                                 unsigned long long* remv_cpu, 
                                        
                                                 int* host_score_indexs,
                                                 long* host_keep_data,
                                                 float* host_boxes,
                                                 int* host_label
                                                 )
{

    
    for (size_t taskIdx = 0; taskIdx < TASK_NUM; taskIdx++){
        std::vector<Box> predBoxs;
        float* reg = static_cast<float*>(buffers.getDeviceBuffer(rpnOutputTensorNames["regName"][taskIdx]));
        float* height = static_cast<float*>(buffers.getDeviceBuffer(rpnOutputTensorNames["heightName"][taskIdx]));
        float* rot = static_cast<float*>(buffers.getDeviceBuffer(rpnOutputTensorNames["rotName"][taskIdx]));
        float* dim = static_cast<float*>(buffers.getDeviceBuffer(rpnOutputTensorNames["dimName"][taskIdx]));
        float* score = static_cast<float*>(buffers.getDeviceBuffer(rpnOutputTensorNames["scoreName"][taskIdx]));
        int32_t* cls = static_cast<int32_t*>(buffers.getDeviceBuffer(rpnOutputTensorNames["clsName"][taskIdx]));
        

        // cudaStream_t stream;
        // GPU_CHECK(cudaStreamCreate(&stream));
        int boxSize = _find_valid_score_num( score, SCORE_THREAHOLD, OUTPUT_H , OUTPUT_W);
        std::cout << " Num boxes before " << boxSize <<"\n";

        _sort_by_key(score, dev_score_indexs, OUTPUT_W * OUTPUT_H);

        boxSize = boxSize > INPUT_NMS_MAX_SIZE ? INPUT_NMS_MAX_SIZE : boxSize;
        // int boxSizeAft = raw_nms_gpu(reg,  height, dim , rot, dev_score_indexs, 
        //                                                 host_keep_data, boxSize,  NMS_THREAHOLD);
        int boxSizeAft = _raw_nms_gpu(reg,  height, dim , rot, dev_score_indexs, 
                                                     host_keep_data, mask_cpu, remv_cpu,  boxSize,  NMS_THREAHOLD);




        boxSizeAft = boxSizeAft > OUTPUT_NMS_MAX_SIZE ? OUTPUT_NMS_MAX_SIZE : boxSizeAft;
        std::cout << " Num boxes after " <<boxSizeAft << "\n";


        // GPU_CHECK(cudaMemcpy(host_keep_data, dev_keep_data, boxSizeAft * sizeof(long), cudaMemcpyDeviceToHost));

        _gather_all(host_boxes, host_label, 
                               reg, height, dim,rot, score, cls, dev_score_indexs, host_keep_data,
                                boxSize,  boxSizeAft );
                                

        GPU_CHECK(cudaMemcpy(host_score_indexs, dev_score_indexs, boxSize * sizeof(int), cudaMemcpyDeviceToHost));
        for(auto i =0; i < boxSizeAft; i++){
            int ii = host_keep_data[i];
            // std::cout <<i<< ", "<<ii<<", \n";
            int idx = host_score_indexs[ii];
            int xIdx = idx % OUTPUT_W;
            int yIdx = idx / OUTPUT_W;
            Box box;

            box.x = (host_boxes[i  + 0 * boxSizeAft] + xIdx) *OUT_SIZE_FACTOR*X_STEP + X_MIN;
            box.y = (host_boxes[i  + 1 * boxSizeAft] + yIdx) * OUT_SIZE_FACTOR*Y_STEP + Y_MIN;
            box.z = host_boxes[i +  2 * boxSizeAft];
            box.l = host_boxes[i +  3 * boxSizeAft];
            box.h = host_boxes[i + 4 * boxSizeAft];
            box.w = host_boxes[i + 5 * boxSizeAft];
            float theta_s =host_boxes[i + 6 * boxSizeAft];
            float theta_c =host_boxes[i + 7 * boxSizeAft];
            box.theta = atan2(theta_s, theta_c);
            box.score  = host_boxes[i + 8 * boxSizeAft];
            box.cls = host_label[i];
            box.velX = idx;
            box.velY = 0;
            predResult.push_back(box);
        }

    }

}




















