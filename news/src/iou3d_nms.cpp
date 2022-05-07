/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/
#include "preprocess.h"

// #include <torch/serialize/tensor.h>
// #include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "iou3d_nms.h"
#include "argsParser.h"
#include <stdio.h>
#include <string.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))



// const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;


void boxesoverlapLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);
void nmsNormalLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);
void rawNmsLauncher(const float *reg, const float* height, const float* dim, const float* rot, const int* indexs, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);





// int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh){

int nms_gpu(const float* boxes_data, long* keep_data, int boxes_num,  float nms_overlap_thresh){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)

    // int boxes_num = boxes.size(0);
    // const float * boxes_data = boxes.data<float>();
    // long * keep_data = keep.data<long>();
    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    GPU_CHECK(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    GPU_CHECK(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);

    unsigned long long remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }

    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );

    return num_to_keep;
}




int raw_nms_gpu(const float* reg,  const float* height, const float* dim , const float* rot, const int* indexs, long* keep_data, int boxes_num,  float nms_overlap_thresh){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)

    // int boxes_num = boxes.size(0);
    // const float * boxes_data = boxes.data<float>();
    // long * keep_data = keep.data<long>();
    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    GPU_CHECK(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    rawNmsLauncher(reg, height, dim, rot, indexs, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    GPU_CHECK(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);

    unsigned long long remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }

    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );

    return num_to_keep;
}



// int nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh){
//     // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
//     // params keep: (N)

//     CHECK_INPUT(boxes);
//     CHECK_CONTIGUOUS(keep);

//     int boxes_num = boxes.size(0);
//     const float * boxes_data = boxes.data<float>();
//     long * keep_data = keep.data<long>();

//     const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

//     unsigned long long *mask_data = NULL;
//     GPU_CHECK(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
//     nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

//     // unsigned long long mask_cpu[boxes_num * col_blocks];
//     // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
//     std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

// //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
//     GPU_CHECK(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
//                            cudaMemcpyDeviceToHost));

//     cudaFree(mask_data);

//     unsigned long long remv_cpu[col_blocks];
//     memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

//     int num_to_keep = 0;

//     for (int i = 0; i < boxes_num; i++){
//         int nblock = i / THREADS_PER_BLOCK_NMS;
//         int inblock = i % THREADS_PER_BLOCK_NMS;

//         if (!(remv_cpu[nblock] & (1ULL << inblock))){
//             keep_data[num_to_keep++] = i;
//             unsigned long long *p = &mask_cpu[0] + i * col_blocks;
//             for (int j = nblock; j < col_blocks; j++){
//                 remv_cpu[j] |= p[j];
//             }
//         }
//     }
//     if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );

//     return num_to_keep;
// }




// int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap){
//     // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
//     // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
//     // params ans_overlap: (N, M)


//     int num_a = boxes_a.size(0);
//     int num_b = boxes_b.size(0);

//     const float * boxes_a_data = boxes_a.data<float>();
//     const float * boxes_b_data = boxes_b.data<float>();
//     float * ans_overlap_data = ans_overlap.data<float>();

//     boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_overlap_data);

//     return 1;
// }

// int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou){
//     // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
//     // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
//     // params ans_overlap: (N, M)

//     int num_a = boxes_a.size(0);
//     int num_b = boxes_b.size(0);

//     const float * boxes_a_data = boxes_a.data<float>();
//     const float * boxes_b_data = boxes_b.data<float>();
//     float * ans_iou_data = ans_iou.data<float>();

//     boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

//     return 1;
// }





