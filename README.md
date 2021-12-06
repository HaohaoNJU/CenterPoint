# An OD project implemented by TensorRT 

The project is a TensorRT version of CenterPoint, an 3D object detection model using center points in the bird eye view.
Code is writen according to the [released paper](https://arxiv.org/abs/2006.11275)

Besides, it is running inference on [WaymoOpenSet](https://waymo.com/intl/en_us/dataset-download-terms) 
![gif](doc/seq0_fp.gif)

You may firstly build the project by executing the following commands:

```
cd /YOUR/PATH/TO/centerpoint/src
mkdir build && cd build
cmake .. && make
./centerpoint
```

To futher learn the detailed documentation, please refer to the following computation graph and doc file.
![graph](doc/computation_graph.png)


# Reference
This project refers to the some of code from :

[CenterPoint](https://github.com/tianweiy/CenterPoint)

[TensorRT](https://github.com/NVIDIA/TensorRT/tree/master)

[CenterPoint-PonintPillars ](https://github.com/CarkusL/CenterPoint)
