#ifndef __CENTERPOINT_POSTPROCESS__
#define __CENTERPOINT_POSTPROCESS__

#include "buffers.h"
#include "common.h"
#include "config.h"
#include <math.h>
#include <stdint.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

struct Box{
    float x;
    float y;
    float z;
    float l;
    float h;
    float w;
    float velX;
    float velY;
    float theta;

    float score;
    int cls;
    bool isDrop; // for nms
};

int rawNmsGpu(const float* reg,  const float* height, const float* dim , const float* rot,
                                     const int* indexs, long* dev_keep_data, unsigned long long* mask_cpu, unsigned long long* remv_gpu,
                                      int boxes_num,  float nms_overlap_thresh);

void sort_by_key(float* keys, int* values,int size) ;

void gather_all(float* host_boxes, int* host_label, 
                                float* reg, float* height, float* dim, float* rot,  float* sorted_score, int32_t* label,  
                                int* dev_indexs, long* host_keep_indexs,  int boxSizeBef, int boxSizeAft) ;

void boxAssignLauncher(float* reg, float* height , float* dim, float*rot, float* boxes, float*score, int* label,  float* out_score, int*out_label,
                                                int* validIndexs ,int boxSize,  int output_h, int output_w) ;
void indexAssignLauncher(int* indexs, int output_h, int output_w) ;
int findValidScoreNum(float* score, float thre, int output_h, int output_w) ;
// void findValidScoreNum(float* score, float thre, int output_h, int output_w, int* box_size); //,  thrust::host_vector<int>  host_box_size);
void postprocessGPU(const samplesCommon::BufferManager& buffers,
                                                 std::vector<Box>& predResult ,
                                                 std::map<std::string, std::vector<std::string>>rpnOutputTensorNames,
                                                 int* dev_score_indexs,
                                                 unsigned long long* mask_cpu,
                                                 unsigned long long* remv_cpu,
                                                 int* host_score_indexs,
                                                 long* host_keep_data,
                                                 float* host_boxes,
                                                 int* host_label);
void postprocess(const samplesCommon::BufferManager& buffers, std::vector<Box>& predResult);
// void postprocessGPU(const samplesCommon::BufferManager& buffers, std::vector<Box>& predResult);
void postTmpSave(const samplesCommon::BufferManager& buffers);

#endif