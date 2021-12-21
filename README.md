# An Lidar Object Detection project implemented by TensorRT 

The project is a TensorRT version of CenterPoint, an 3D object detection model using center points in the bird eye view.
Code is writen according to the [released paper](https://arxiv.org/abs/2006.11275)

Besides, it is running inference on [WaymoOpenSet](https://waymo.com/intl/en_us/dataset-download-terms) 
![gif](doc/seq0_fp.gif)

# Setup

The project has beed tested on *Ubuntu18.04* and *Ubuntu20.04*, 
It mainly relys on TensorRT and cuda as 3rd-party package,  with the following versions respectively:

*vTensorRT : 8.0.1.6*

*vCuda : 11.3*

Note that this project does not rely on *PCL* and *Boost* by now, however they may be used in the future and has been writen in CMakeLists.txt.

After installation, you may then build the project by executing the following commands:

```
cd /YOUR/PATH/TO/centerpoint/src
mkdir build && cd build
cmake .. && make
./centerpoint
```
By default, the project load the serialized engine file to do inference, the engine files are created by the onnx files we provided and is set to float16.
You can also build from onnx file by setting `params.load_engine = false` in samplecenterpoint.cpp and provide the onnx file path, in that way, you may control whether to use fp16 or fp32.

# What has been done?
To futher learn the detailed documentation, please refer to the following computation graph and [doc file](doc/CenterPointTRT.doc).
![graph](doc/computation_graph.png)

# Computation Speed 
Acceleration is the main aim we want to archieve, therefore we do most of computation(including preprocess & postprocess) on GPU, 
the below table gives the average computation speed (by millisecond) of every computation module, it is tested on RTX3080, with all the 39987 waymo validation samples. As illustrated above, the engine is set to float16, and eveluation metric shows no difference on fp32 or fp16.

|Preprocess|PfeInfer|ScatterInfer|RpnInfer|Postprocess|
|---|---|---|---|---|
|1.61|5.88|0.17|6.89|2.37|

# Acknowledgements
This project refers to some of code from :

[CenterPoint](https://github.com/tianweiy/CenterPoint)

[TensorRT](https://github.com/NVIDIA/TensorRT/tree/master)

[CenterPoint-PointPillars ](https://github.com/CarkusL/CenterPoint)
