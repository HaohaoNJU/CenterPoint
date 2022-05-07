#ifndef __CENTERNET_CONFIG_H__
#define __CENTERNET_CONFIG_H__

// ========================================WAYMO CENTERPOINT CONFIG======================================== 
// point size
#define MAX_POINTS 220000
#define POINT_DIM 5

// pillar size 
#define X_STEP 0.32f
#define Y_STEP 0.32f
#define X_MIN -74.88f
#define X_MAX 74.88f
#define Y_MIN -74.88f
#define Y_MAX 74.88f
#define Z_MIN -2.0f
#define Z_MAX 4.0f

#define X_CENTER_MIN -80.0f
#define X_CENTER_MAX 80.0f
#define Y_CENTER_MIN -80.0f
#define Y_CENTER_MAX 80.0f
#define Z_CENTER_MIN -10.0f
#define Z_CENTER_MAX 10.0f

#define PI 3.141592653f
// paramerters for preprocess
#define BEV_W 468
#define BEV_H 468
#define MAX_PILLARS 32000 //20000 //32000
#define MAX_PIONT_IN_PILLARS 20
#define FEATURE_NUM 10
#define PFE_OUTPUT_DIM 64
#define THREAD_NUM 4
// paramerters for postprocess
#define SCORE_THREAHOLD 0.1f
#define NMS_THREAHOLD 0.7f
#define INPUT_NMS_MAX_SIZE 4096
#define OUTPUT_NMS_MAX_SIZE 500
// #define THREADS_PER_BLOCK_NMS  sizeof(unsigned long long) * 8
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

// OUT_SIZE_FACTOR * OUTPUT_H  * Y_STEP = Y_MAX - Y_MIN
#define OUT_SIZE_FACTOR 1.0f    

#define TASK_NUM 1
#define REG_CHANNEL 2
#define HEIGHT_CHANNEL 1
#define ROT_CHANNEL 2
// #define VEL_CHANNEL 2 //don't defined in waymo
#define DIM_CHANNEL 3

// spatial output size of rpn 
#define OUTPUT_H 468  
#define OUTPUT_W 468
#endif






// ========================================NUSCENES CENTERPOINT CONFIG======================================== 

// // pillar size 
// #define X_STEP 0.2f
// #define Y_STEP 0.2f
// #define X_MIN -51.2f
// #define X_MAX 51.2f
// #define Y_MIN -51.2f
// #define Y_MAX 51.2f
// #define Z_MIN -5.0f
// #define Z_MAX 3.0f
// #define PI 3.141592653f
// // paramerters for preprocess
// #define BEV_W 512
// #define BEV_H 512
// #define MAX_PILLARS 30000
// #define MAX_PIONT_IN_PILLARS 20
// #define FEATURE_NUM 10
// #define THREAD_NUM 2
// // paramerters for postprocess
// #define SCORE_THREAHOLD 0.1f
// #define NMS_THREAHOLD 0.2f
// #define INPUT_NMS_MAX_SIZE 1000
// #define OUT_SIZE_FACTOR 4.0f
// #define TASK_NUM 6
// #define REG_CHANNEL 2
// #define HEIGHT_CHANNEL 1
// #define ROT_CHANNEL 2
// #define VEL_CHANNEL 2
// #define DIM_CHANNEL 3
// #define OUTPUT_H 128
// #define OUTPUT_W 128
// #endif

















