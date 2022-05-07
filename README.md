# An Lidar Object Detection project implemented by TensorRT 

The project is a TensorRT version of CenterPoint, an 3D object detection model using center points in the bird eye view.
Code is written according to the [project](https://github.com/tianweiy/CenterPoint.git)

Besides, it is running inference on [WaymoOpenSet](https://waymo.com/intl/en_us/dataset-download-terms) 


# Setup

The project has been tested on *Ubuntu18.04* and *Ubuntu20.04*, 
It mainly relies on TensorRT and cuda as 3rd-party package,  with the following versions respectively:

*vTensorRT : 8.0.1.6*

*vCuda : 11.3*

This project has provided the baseline onnx models trained with [this config](https://github.com/tianweiy/CenterPoint/blob/master/configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py) in `models`. If you want to export your own models, we assume you have had [CenterPoint](https://github.com/tianweiy/CenterPoint.git) project installed, you can setup local `det3d` environment 

```
cd /PATH/TO/centerpoint/tools 
bash setup3.sh
```

# Preperation 

###  Export as onnx models
To export your own models, you can run
```
python3 export_onnx.py \
--config waymo_centerpoint_pp_two_pfn_stride1_3x.py \
--ckpt your_model.pth \
--pfe_save_path pfe.onnx \
--rpn_save_path rpn.onnx
```
Here we extract two pure nn models from the whole computation graph---`pfe` and `rpn`, this is to make it easier for trt to optimize its inference engine, 
and we use cuda to connect these nn engines.

###  Generate TensorRT serialized engines
Actually you can directly create trt engines from onnx models and skip this step, however a more ideal way is to load your previously saved serialize engine files.

You can run 
```
python3 create_engine.py \
--config waymo_centerpoint_pp_two_pfn_stride1_3x.py \
--pfe_onnx_path pfe.onnx \
--rpn_onnx_path rpn.onnx \
--pfe_engine_path pfe_fp.engine \
--rpn_engine_path rpn_fp.engine
```
By default this will generate fp16-engine files.

### Work with int8


###  Run inference 

## 4. Online Tracking and Visualization

After installation, you may then build the project by executing the following commands:

```
cd /YOUR/PATH/TO/centerpoint
mkdir centerpoint_pp_baseline_score0.1_nms0.7 && cd src
mkdir build && cd build
cmake .. && make
./centerpoint
```
By default, the project loads the serialized engine files to do inference, and the engine files are created by the onnx files we provided and are set as float16.
You can also build from onnx files by setting `params.load_engine = false` in samplecenterpoint.cpp and provide the onnx file path. In that way, you may decide whether to use fp16 or fp32.

# What has been done?
To futher learn the detailed documentation, please refer to the following computation graph and [doc file](doc/CenterPointTRT.doc).
![graph](doc/computation_graph.png)

# Computation Speed 
Acceleration is the main aim we want to archieve, and therefore we do most of computation(including preprocess & postprocess) on GPU. 
The below table gives the average computation speed (by millisecond) of every computation module, and it is tested on RTX3080, with all the 39987 waymo validation samples. As illustrated above, the engine is set to float16, and eveluation metric shows no difference on fp32 or fp16.

|Preprocess|PfeInfer|ScatterInfer|RpnInfer|Postprocess|
|---|---|---|---|---|
|1.61|5.88|0.17|6.89|2.37|

Detection result shows below:
![gif](doc/seq0_fp.gif)

# Acknowledgements
This project refers to some codes from:

[CenterPoint](https://github.com/tianweiy/CenterPoint)

[TensorRT](https://github.com/NVIDIA/TensorRT/tree/master)

[CenterPoint-PointPillars ](https://github.com/CarkusL/CenterPoint)
