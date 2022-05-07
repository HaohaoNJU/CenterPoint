#include "centerpoint.h"
#include "preprocess.h"


bool CenterPoint::engineInitlization()
 {
        sample::gLogInfo << "Building pfe engine . . .  "<< std::endl;
        mEngine = mParams.load_engine ? buildFromSerializedEngine(mParams.pfeSerializedEnginePath) : build(mParams.pfeOnnxFilePath);
        sample::gLogInfo << "Building rpn engine . . .  "<< std::endl;
        mEngineRPN = mParams.load_engine ? buildFromSerializedEngine(mParams.rpnSerializedEnginePath) : build(mParams.rpnOnnxFilePath);
        sample::gLogInfo << "All has Built !  "<< std::endl;
        return true;
}

std::shared_ptr<nvinfer1::ICudaEngine> CenterPoint::buildFromSerializedEngine(std::string serializedEngineFile) 
{

     std::vector<char> trtModelStream_;
     size_t size{0};
     std::ifstream file(serializedEngineFile, std::ios::binary);
     if (file.good()) 
    {
         file.seekg(0, file.end);
         size = file.tellg();
         file.seekg(0,file.beg);
         trtModelStream_.resize(size);
         file.read(trtModelStream_.data(), size);
         file.close() ;
     }
     else 
     {
        sample::gLogError<< " Failed to read serialized engine ! " << std::endl;
        return nullptr;
     }
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if(!runtime) { sample::gLogError << "Failed to create runtime \n"; return nullptr;}
    sample::gLogInfo<<"Create ICudaEngine  !" << std::endl;
    std::shared_ptr<nvinfer1::ICudaEngine>  engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(trtModelStream_.data(), size), 
        samplesCommon::InferDeleter());

    if (!engine)
    {
        sample::gLogError << "Failed to create engine \n";
        return nullptr;
    }

    return engine;
}

std::shared_ptr<nvinfer1::ICudaEngine>  CenterPoint::build(std::string  onnxFilePath)
{
    // We assumed that nvinfer1::createInferBuilder is droped in TRT 8.0 or above
    
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    { 
        sample::gLogError<< "Builder not created !" << std::endl;
        return nullptr;
    }
   



    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        sample::gLogError<< "Network not created ! " << std::endl;
        return nullptr;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        sample::gLogError<< "Config not created ! " << std::endl;
        return nullptr;
    }
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        sample::gLogError<< "Parser not created ! " << std::endl;
        return nullptr;
    }
    sample::gLogInfo<<"ConstructNetwork !" << std::endl;
    
    cudaEvent_t  start, end;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,stream);    

    auto constructed = constructNetwork(builder, network, config, parser,onnxFilePath);


    if (!constructed)
    {
        return nullptr;
    }

    ///////////////////////////////////////////////////////////////////BUILD ENGINE FROM FILE//////////////////////////////////////////////////////////////////////////////

    // std::shared_ptr<nvinfer1::ICudaEngine> engine = std::shared_ptr<nvinfer1::ICudaEngine>(
    //     builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    
    // cuda stream used to profiling the builder
    auto profileStream = samplesCommon::makeCudaStream();
    if(!profileStream) {
        sample::gLogError<<"Failed to create a profile stream !\n";
        return  nullptr;
    }
    config->setProfileStream(*profileStream);    

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {sample::gLogError << "Failed to create IHostMemory plan \n";return  nullptr;}


    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if(!runtime) { sample::gLogError << "Failed to create runtime \n"; return nullptr;}
    sample::gLogInfo<<"Create ICudaEngine  !" << std::endl;
    std::shared_ptr<nvinfer1::ICudaEngine> engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    if (!engine)
    {
        sample::gLogError << "Failed to create engine \n";
        return nullptr;
    }

    sample::gLogInfo << "getNbInputs: " << network->getNbInputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs: " << network->getNbOutputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs Name: " << network->getOutput(0)->getName() << " \n" << std::endl;

    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();
    return engine;
}



bool CenterPoint::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser,
    std::string  onnxFilePath)
{   
    auto parsed = parser->parseFromFile(
        // locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        // params.onnxFileName.c_str(),
        onnxFilePath.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));

        // ILogger::Severity::kWARNING);
    if (!parsed)
    {
        sample::gLogError<< "Onnx model cannot be parsed ! " << std::endl;
        return false;
    }
    builder->setMaxBatchSize(BATCH_SIZE_);
    config->setMaxWorkspaceSize(5_GiB); //8_GiB);
    if (mParams.fp16)
        config->setFlag(BuilderFlag::kFP16);
    if (mParams.dlaCore >=0 ){
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    sample::gLogInfo << "Deep Learning Acclerator (DLA) was enabled . \n";
    }
    return true;
}


//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool CenterPoint::infer()
{
    // Create RAII buffer manager object
    sample::gLogInfo << "Creating pfe context " << std::endl;
    samplesCommon::BufferManager buffers(mEngine);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    sample::gLogInfo << "Creating rpn context " << std::endl;
    samplesCommon::BufferManager buffersRPN(mEngineRPN);
    auto contextRPN = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngineRPN->createExecutionContext());

    if (!context || !contextRPN)
    {
           sample::gLogError<< "Failed to create context " << std::endl;
        return false;
    }

    // mParams.inputTensorNames :  [ voxels,  num_voxels, coords ]
    float* devicePillars = static_cast<float*>(buffers.getDeviceBuffer(mParams.pfeInputTensorNames[0]));

    // int hostIndex[MAX_PILLARS] = {-1};

    int voxelNum = 0;
    void* inputPointBuf = nullptr;



    // create event  object , which are used time computing
    cudaEvent_t start, stop;
    float pre_time = 0;
    float pfe_time = 0;
    float scatter_time = 0;
    float rpn_time  = 0;
    float post_time = 0;
    
    float totalPreprocessDur = 0;
    float totalPostprocessDur = 0;
    float totalScatterDur =0 ;
    float totalPfeDur = 0;
    float totalRpnDur = 0;

    int fileSize = mParams.filePaths.size();
    // int fileSize = 1;

    if (!fileSize) {
        sample::gLogError<< "No Bin File Was Found ! " << std::endl;
        return false;
    }

    // For Loop Every Pcl Bin 
    // for(auto idx = 0; idx < filePath.size(); idx++){
     for(auto idx = 0; idx < fileSize; idx++){
        std::cout << "===========FilePath[" << idx <<"/"<<fileSize<<"]:" << mParams.filePaths[idx] <<"=============="<< std::endl;
        
        int pointNum = 0;
        if (!processInput(inputPointBuf, mParams.filePaths[idx], pointNum))
        {
            return false;
        }

        // Create cuda stream to profile this infer pipline 
        cudaStream_t stream;
        GPU_CHECK(cudaStreamCreate(&stream));

        float* points = static_cast<float*>(inputPointBuf);
        std::vector<Box> predResult;


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        //  Doing preprocess 
        // preprocess(points, hostPillars, hostIndex, pointNum,POINT_DIM);
        GPU_CHECK(cudaMemcpy(dev_points, points, pointNum * POINT_DIM * sizeof(float), cudaMemcpyHostToDevice));
        preprocessGPU(dev_points, devicePillars,deviceIndices, 
        _PMask, _PBEVIdxs,  _PPointNumAssigned,  _BEVVoxelIdx, _VPointSum,  _VRange,  _VPointNum,
         pointNum, POINT_DIM);

        
        // Memcpy from host input buffers to device input buffers
        // buffers.copyInputToDeviceAsync(stream);
        // buffers.copyInputToDevice();


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&pre_time, start, stop);

        // Doing inference 


        cudaEventRecord(start);
        bool status = context->executeV2(buffers.getDeviceBindings().data());
        // auto  status = context->enqueue(BATCH_SIZE_, buffers.getDeviceBindings().data(), stream, nullptr);


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
         cudaEventElapsedTime(&pfe_time, start, stop);
        cudaEventRecord(start);
        if (!status)
        {
            sample::gLogError<< "Error with pfe contex execution ! " << std::endl;
            return false;
        }

        // copy coordinates from host to device 
        // GPU_CHECK(cudaMemcpyAsync(deviceIndices, hostIndex, MAX_PILLARS * sizeof(int), cudaMemcpyHostToDevice, stream));
        // GPU_CHECK(cudaMemcpy(deviceIndices, hostIndex, MAX_PILLARS * sizeof(int), cudaMemcpyHostToDevice));

        //  cast value type on the GPU device 
        dev_scattered_feature_ = static_cast<float*>(buffersRPN.getDeviceBuffer(mParams.rpnInputTensorNames[0]));

        // reset scattered feature to zero . 
        GPU_CHECK(cudaMemset(dev_scattered_feature_, 0 ,  PFE_OUTPUT_DIM * BEV_W * BEV_H * sizeof(float)));
        
        scatter_cuda_ptr_->doScatterCuda(MAX_PILLARS, deviceIndices,static_cast<float*>(buffers.getDeviceBuffer(mParams.pfeOutputTensorNames[0])), 
                                                                //   static_cast<float*>(buffersRPN.getDeviceBuffer(mParamsRPN.inputTensorNames[0]) )) ;
                                                                dev_scattered_feature_);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
         cudaEventElapsedTime(&scatter_time, start, stop);
        cudaEventRecord(start);
        // status = contextRPN->enqueue( BATCH_SIZE_,buffersRPN.getDeviceBindings().data(), stream, nullptr);
        status = contextRPN->executeV2( buffersRPN.getDeviceBindings().data());
        if (!status)
        {
            sample::gLogError<< "Error with rpn contex execution ! " << std::endl;
            return false;
        }
        // Copying outputs from device to host 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&rpn_time, start, stop);
        
        // Doing postprocess 
        predResult.clear();
        cudaEventRecord(start);

        // buffersRPN.copyOutputToHostAsync(stream);
        // buffersRPN.copyOutputToHost();


        postprocessGPU(buffersRPN, predResult, mParams.rpnOutputTensorNames,
                                                dev_score_indexs_,
                                                mask_cpu,
                                                remv_cpu,
                                                host_score_indexs_,
                                                host_keep_data_,
                                                host_boxes_,
                                                host_label_);


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&post_time, start, stop);

        totalPreprocessDur += pre_time;
        totalScatterDur += scatter_time;
        totalPfeDur += pfe_time;
        totalRpnDur += rpn_time;
        totalPostprocessDur += post_time;

        saveOutput(predResult, mParams.filePaths[idx], mParams.savePath);
        free(points);  

        // release the stream and  the buffers
        cudaStreamDestroy(stream);
    }
    sample::gLogInfo << "Average PreProcess Time: " << totalPreprocessDur / fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average PfeInfer Time: " << totalPfeDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average ScatterInfer Time: " << totalScatterDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average RpnInfer  Time: " << totalRpnDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average PostProcess Time: " << totalPostprocessDur /  fileSize<< " ms"<< std::endl;

    return true;
}

/* There is a bug. 
 * If I change void to bool, the "for (size_t idx = 0; idx < mEngine->getNbBindings(); idx++)" loop will not stop.
 */

void CenterPoint::saveOutput(std::vector<Box>& predResult, std::string& inputFileName,  std::string savePath)
{
    
    std::string::size_type pos = inputFileName.find_last_of("/");
    std::string outputFilePath = savePath + "/" +  inputFileName.substr(pos) + ".txt";


    ofstream resultFile;

    resultFile.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    try {
        resultFile.open(outputFilePath);
        for (size_t idx = 0; idx < predResult.size(); idx++){
                resultFile << predResult[idx].x << " " << predResult[idx].y << " " << predResult[idx].z << " "<< \
                predResult[idx].l << " " << predResult[idx].h << " " << predResult[idx].w << " " << predResult[idx].velX \
                << " " << predResult[idx].velY << " " << predResult[idx].theta << " " << predResult[idx].score << \ 
                " "<< predResult[idx].cls << std::endl;
        }
        resultFile.close();
    }
    catch (std::ifstream::failure e) {
        sample::gLogError << "Open File: " << outputFilePath << " Falied"<< std::endl;
    }
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool CenterPoint::processInput(void*& inputPointBuf, std::string& pointFilePath, int& pointNum)
{

    bool ret = readBinFile(pointFilePath, inputPointBuf, pointNum,  POINT_DIM);
    std::cout << "Success to read and Point Num  Is: " << pointNum << std::endl;
    if(!ret){
        sample::gLogError << "Error read point file: " << pointFilePath<< std::endl;
        free(inputPointBuf);
        return ret;
    }
    return ret;
}
