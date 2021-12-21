

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include "preprocess.h"
#include "postprocess.h"
#include "scatter_cuda.h"

// below  head files are defined in TensorRt/samples/common
#include "EntropyCalibrator.h"
#include "BatchStream.h"

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))



struct Params{
    std::string pfeOnnxFilePath = "/home/wanghao/Desktop/projects/notebooks/pfe_baseline32000.onnx";
    std::string rpnOnnxFilePath = "/home/wanghao/Desktop/projects/notebooks/rpn_baseline.onnx";
    std::string pfeSerializedEnginePath = "";
    std::string rpnSerializedEnginePath = "";

    // Input Output Names
    std::vector<std::string> pfeInputTensorNames;
    std::vector<std::string> rpnInputTensorNames;
    std::vector<std::string> pfeOutputTensorNames;
    std::map<std::string, std::vector<std::string>> rpnOutputTensorNames;

    // Input Output Paths
    std::string savePath ;
    std::vector<std::string>  filePaths;
    
    // Attrs
    int dlaCore = -1;
    bool fp16 = false;
    bool int8 = false;
    bool load_engine = false;
    int batch_size = 1;
};

class CenterPoint
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    CenterPoint(const Params params)
        : mParams(params)
        ,BATCH_SIZE_(params.batch_size)
        , mEngine(nullptr)
        ,mEngineRPN(nullptr)
    {

        //const int NUM_THREADS, const int MAX_NUM_PILLARS, const int GRID_X_SIZE, const int GRID_Y_SIZE):
        scatter_cuda_ptr_.reset(new ScatterCuda(PFE_OUTPUT_DIM, PFE_OUTPUT_DIM, BEV_W, BEV_H ));
        // mallocate a global memory for pointer
        // GPU_CHECK(cudaMalloc((void**)&dev_scattered_feature_,PFE_OUTPUT_DIM * BEV_H * BEV_W * sizeof(float)));

        GPU_CHECK(cudaMalloc((void**)&dev_points, MAX_POINTS * POINT_DIM * sizeof(float)));
        GPU_CHECK(cudaMemset(dev_points,0, MAX_POINTS * POINT_DIM * sizeof(float)));

        GPU_CHECK(cudaMalloc((void**)&deviceIndices,MAX_PILLARS * sizeof(int)));
        GPU_CHECK(cudaMemset(deviceIndices,0,MAX_PILLARS * sizeof(int)));

        /**
         * @brief : Create and Init Variables for PreProcess
         * 
         */
        GPU_CHECK(cudaMalloc((void**)& _PBEVIdxs, MAX_POINTS * sizeof(int)));
        GPU_CHECK(cudaMalloc((void**)& _PPointNumAssigned, MAX_POINTS * sizeof(int)));
        GPU_CHECK(cudaMalloc((void**)& _PMask, MAX_POINTS * sizeof(bool)));
        GPU_CHECK(cudaMalloc((void**)& _BEVVoxelIdx, BEV_H * BEV_W * sizeof(int)));

        GPU_CHECK(cudaMemset(_PBEVIdxs, 0, MAX_POINTS * sizeof(int)));
        GPU_CHECK(cudaMemset(_PPointNumAssigned, 0, MAX_POINTS * sizeof(int)));
        GPU_CHECK(cudaMemset(_PMask, 0, MAX_POINTS * sizeof(bool)));
        GPU_CHECK(cudaMemset(_BEVVoxelIdx, 0, BEV_H * BEV_W * sizeof(int)));

        GPU_CHECK(cudaMalloc((void**)&_VPointSum, MAX_PILLARS * 3 *sizeof(float)));
        GPU_CHECK(cudaMalloc((void**)&_VRange, MAX_PILLARS * sizeof(int)));
        GPU_CHECK(cudaMalloc((void**)&_VPointNum, MAX_PILLARS * sizeof(int)));


        GPU_CHECK(cudaMemset(_VRange,0, MAX_PILLARS * sizeof(int)));
        GPU_CHECK(cudaMemset(_VPointSum, 0, MAX_PILLARS * 3 * sizeof(float)));

        /**
         * @brief : Create and Init Variables for PostProcess
         * 
         */
        GPU_CHECK(cudaMalloc((void**)&dev_score_indexs_, OUTPUT_W * OUTPUT_H * sizeof(int)));
        GPU_CHECK(cudaMemset(dev_score_indexs_, -1 , OUTPUT_W * OUTPUT_H * sizeof(int)));

        // GPU_CHECK(cudaMalloc((void**)& dev_keep_data_, INPUT_NMS_MAX_SIZE * DIV_UP (INPUT_NMS_MAX_SIZE ,INPUT_NMS_MAX_SIZE) * sizeof(unsigned long long)));
        // GPU_CHECK(cudaMemset(dev_keep_data_, -1 ,  INPUT_NMS_MAX_SIZE * DIV_UP (INPUT_NMS_MAX_SIZE ,INPUT_NMS_MAX_SIZE)  * sizeof(unsigned long long)));

        GPU_CHECK(cudaMallocHost((void**)& mask_cpu, INPUT_NMS_MAX_SIZE * DIVUP (INPUT_NMS_MAX_SIZE ,THREADS_PER_BLOCK_NMS) * sizeof(unsigned long long)));
        GPU_CHECK(cudaMemset(mask_cpu, 0 ,  INPUT_NMS_MAX_SIZE * DIVUP (INPUT_NMS_MAX_SIZE ,THREADS_PER_BLOCK_NMS) * sizeof(unsigned long long)));

        GPU_CHECK(cudaMallocHost((void**)& remv_cpu, THREADS_PER_BLOCK_NMS * sizeof(unsigned long long)));
        GPU_CHECK(cudaMemset(remv_cpu, 0 ,  THREADS_PER_BLOCK_NMS  * sizeof(unsigned long long)));

        GPU_CHECK(cudaMallocHost((void**)&host_score_indexs_, OUTPUT_W * OUTPUT_H  * sizeof(int)));
        GPU_CHECK(cudaMemset(host_score_indexs_, -1, OUTPUT_W * OUTPUT_H  * sizeof(int)));

        GPU_CHECK(cudaMallocHost((void**)&host_keep_data_, INPUT_NMS_MAX_SIZE * sizeof(long)));
        GPU_CHECK(cudaMemset(host_keep_data_, -1, INPUT_NMS_MAX_SIZE * sizeof(long)));

        GPU_CHECK(cudaMallocHost((void**)&host_boxes_, OUTPUT_NMS_MAX_SIZE * 9 * sizeof(float)));
        GPU_CHECK(cudaMemset(host_boxes_, 0 ,  OUTPUT_NMS_MAX_SIZE * 9 * sizeof(float)));

        GPU_CHECK(cudaMallocHost((void**)&host_label_, OUTPUT_NMS_MAX_SIZE * sizeof(int)));
        GPU_CHECK(cudaMemset(host_label_, -1, OUTPUT_NMS_MAX_SIZE * sizeof(int)));

    }

    ~CenterPoint() {
    // Free host pointers
    // Free global pointers 
    std::cout << "Free Variables . \n";
    GPU_CHECK(cudaFree(deviceIndices));
    GPU_CHECK(cudaFree(dev_points));
    GPU_CHECK(cudaFree(dev_score_indexs_));
    // GPU_CHECK(cudaFree(dev_scattered_feature_));
    // GPU_CHECK(cudaFree(dev_keep_data_));

    GPU_CHECK(cudaFree( _PBEVIdxs)); 
    GPU_CHECK(cudaFree( _PPointNumAssigned));
    GPU_CHECK(cudaFree( _PMask));
    GPU_CHECK(cudaFree( _BEVVoxelIdx)); // H * W
    GPU_CHECK(cudaFree( _VPointSum));
    GPU_CHECK(cudaFree( _VRange));
    GPU_CHECK(cudaFree( _VPointNum));


    GPU_CHECK(cudaFreeHost(host_keep_data_));
    GPU_CHECK(cudaFreeHost(host_boxes_));
    GPU_CHECK(cudaFreeHost(host_label_));
    GPU_CHECK(cudaFreeHost(host_score_indexs_));
    GPU_CHECK(cudaFreeHost(remv_cpu));
    GPU_CHECK(cudaFreeHost(mask_cpu));


    // // Free engine 
    // std::cout << "Free PFE Engine  .\n";
    // mEngine->destroy();
    // std::cout << "Free RPN Engine  .\n";
    // mEngineRPN->destroy();
    }

    std::shared_ptr<nvinfer1::ICudaEngine>  build( std::string onnxFilePath);
    std::shared_ptr<nvinfer1::ICudaEngine>  buildFromSerializedEngine(std::string serializedEngineFile);
    bool infer();
    bool engineInitlization();
    


private:
    // device pointers 
    float* dev_scattered_feature_;
    float* dev_points ;
    int* deviceIndices;
    int* dev_score_indexs_;
    long* dev_keep_data_;
    SampleUniquePtr<ScatterCuda> scatter_cuda_ptr_;

    // device pointers for preprocess
    int* _PBEVIdxs; 
    int* _PPointNumAssigned;
    bool* _PMask;
    int* _BEVVoxelIdx; // H * W
    float* _VPointSum;
    int* _VRange;
    int* _VPointNum;
    

    // host  variables for post process
    long* host_keep_data_;
    float* host_boxes_;
    int* host_label_;
    int* host_score_indexs_;
    unsigned long long* mask_cpu;
    unsigned long long* remv_cpu;

    Params mParams;
    int BATCH_SIZE_ = 1;
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::ICudaEngine> mEngineRPN;
    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser,
       std::string onnxFilePath);
    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(void*& points, std::string& pointFilePath, int& pointNum);
    //!
    //! \brief Classifies digits and verify result
    //!
    void saveOutput(std::vector<Box>& predResult, std::string& inputFileName, std::string savePath);
};




