/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include "preprocess.h"
#include "postprocess.h"
#include "scatter_cuda.h"
#include "centerpoint.h"
#include "utils.h"

const std::string gSampleName = "TensorRT.sample_onnx_centerpoint";

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./centerpoint [-h or --help]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--filePath       Specify path to a data directory. "
              << std::endl;
    std::cout << "--savePath       Specify path to a directory you want save detection results."
              << std::endl;

    std::cout << "--loadEngine       Load from serialized engine files or from onnx files, provide this argument only when you want to create "
    "engine from serialized engine files you previously generated(and provide paths to engine files), or you will need to provide paths to onnx files. "
              << std::endl;   

    std::cout << "--pfeOnnxPath       Specify path to pfe onnx model. This option can be used when you want to create engine from onnx file. "
              << std::endl;
    std::cout << "--rpnOnnxPath       Specify path to rpn onnx model. This option can be used when you want to create engine from onnx file. "
              << std::endl;      
    std::cout << "--pfeEnginePath       Specify path to pfe engine model. This option can be used when you want to create engine from serialized engine file you previously generated. "
              << std::endl;
    std::cout << "--rpnEnginePath       Specify path to rpn engine model. This option can be used when you want to create engine from serialized engine file you previously generated.  "
              << std::endl;   

    std::cout << "--fp16       Provide this argument only when you want  to do inference on fp16 mode, note that this config is only valid when you create engine from onnx files. "
              << std::endl;   

    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform, by default it's set -1."
              << std::endl;
    
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);



    ///////////////////////////////////////////////////////////////PARAM INITIALIZATION///////////////////////////////////////////////////////////////
    Params params;
    // initialize sample parameters 
    params.pfeOnnxFilePath =  args.pfeOnnxPath;
    params.rpnOnnxFilePath =  args.rpnOnnxPath;
    params.pfeSerializedEnginePath = args.pfeEnginePath;
    params.rpnSerializedEnginePath = args.rpnEnginePath;
    params.savePath = args.savePath;
    params.filePaths=glob(args.filePath + "/seq_*.bin");
    params.fp16 = args.runInFp16;
    params.load_engine = args.loadEngine;

    // Input Output Names, according to TASK_NUM
    params.pfeInputTensorNames.push_back("input.1");
    params.rpnInputTensorNames.push_back("input.1");
    params.pfeOutputTensorNames.push_back("47");

    params.rpnOutputTensorNames["regName"]  = {"246"};
    params.rpnOutputTensorNames["rotName"] = {"258"};
    params.rpnOutputTensorNames["heightName"]={"250"};
    params.rpnOutputTensorNames["dimName"] = {"264"};
    params.rpnOutputTensorNames["scoreName"] = {"265"};
    params.rpnOutputTensorNames["clsName"] = {"266"};


    // Attrs
    params.dlaCore = args.useDLACore;
    params.batch_size = 1;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // std::string savePath = "/home/wanghao/Desktop/projects/notebooks/centerpoint_output_cpp" ;
    CenterPoint sample(params);
    sample::gLogInfo << "Building and running a GPU inference engine for CenterPoint" << std::endl;
    if (!sample.engineInitlization())
    {
        sample::gLogInfo << "sample build error  " << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }
    
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogger.reportPass(sampleTest);
    return 1;
}







