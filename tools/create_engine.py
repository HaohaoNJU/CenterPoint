import glob
# import onnxruntime
import torch
import os
import sys
import numpy as np
import pcdet
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import matplotlib.pyplot as plt
from time import time
import argparse 
from det3d.torchie import Config


def parse_args():
    parser = argparse.ArgumentParser(description="create engine files ")
    parser.add_argument("--config", help="train config file path",type=str, default='waymo_centerpoint_pp_two_pfn_stride1_3x.py')

    parser.add_argument("--pfe_onnx_path", help="the dir to load pfe  onnx",type = str, default = "pfe.onnx")
    parser.add_argument("--rpn_onnx_path", help="the dir to load rpn  onnx",type = str, default = "rpn.onnx")

    parser.add_argument("--pfe_engine_path", help="the dir to save pfe  engine",type = str, default = "pfe.engine")
    parser.add_argument("--rpn_engine_path", help="the dir to save rpn  engine",type = str, default = "rpn.engine")
    
    parser.add_argument("--quant", action= 'store_true', help='whether to make quantilization ! ')
    parser.add_argument("--minmax_calib", action= 'store_true', help='whether to make MinMaxCalibration, by default we use EntropyCalib! ')

    parser.add_argument("--calib_file_path", help="the dir to calibration files, only config when `quant` is enabled. ",type = str)
    parser.add_argument("--calib_batch_size", type = int , default = 1, help = "batch size for calibration.")
    
    args = parser.parse_args()
    return args

args = parse_args()

class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    
    def __init__(self, datas, cache_file="calib_cache.bin", batch_size=1,shape = [32000,20,10]):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.shape = shape
        self.shape.insert(0,batch_size)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        assert isinstance(datas, list) and len(datas), "datas should be a type of `list`and should not be empty. "
        # if isinstance(datas[0], str) :   self.read_cache = False
        # elif isinstance(datas[0], np.ndarray) : self.read_cache = True
        # else: raise TypeError("Can't recognize calibration data types.")
        self.datas = datas
        self.data = self.read_data(0)

        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        # print("alloc size " , self.data[0].nbytes * self.batch_size)
        self.device_input = cuda.mem_alloc(self.data.nbytes * self.batch_size)
    def read_data(self,idx):
        data = np.fromfile(self.datas[idx],dtype = np.float32)
        return data
        
    def get_batch_size(self):
        return self.batch_size
    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.datas):
            return None
        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}/{}".format(current_batch,  len(self.datas)//self.batch_size ))
        batch = []
        for i in range(self.batch_size):
            sample = self.read_data(self.current_index + i)
            batch.append(sample)
        batch = np.stack(batch,axis = 0)
        batch = np.ascontiguousarray(batch.reshape(*self.shape)).ravel()
        #batch = self.read_data(self.current_index )
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()
        pass
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)



class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    
    def __init__(self, datas, cache_file="calib_cache.bin", batch_size=1,shape = [32000,20,10]):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.shape = shape
        self.shape.insert(0,batch_size)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        assert isinstance(datas, list) and len(datas), "datas should be a type of `list`and should not be empty. "
        # if isinstance(datas[0], str) :   self.read_cache = False
        # elif isinstance(datas[0], np.ndarray) : self.read_cache = True
        # else: raise TypeError("Can't recognize calibration data types.")
        self.datas = datas
        self.data = self.read_data(0)

        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        # print("alloc size " , self.data[0].nbytes * self.batch_size)
        self.device_input = cuda.mem_alloc(self.data.nbytes * self.batch_size)
    def read_data(self,idx):
        data = np.fromfile(self.datas[idx],dtype = np.float32)
        return data
        
    def get_batch_size(self):
        return self.batch_size
    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.datas):
            return None
        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}/{}".format(current_batch,  len(self.datas)//self.batch_size ))
        batch = []
        for i in range(self.batch_size):
            sample = self.read_data(self.current_index + i)
            batch.append(sample)
        batch = np.stack(batch,axis = 0)
        batch = np.ascontiguousarray(batch.reshape(*self.shape)).ravel()
        #batch = self.read_data(self.current_index )
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()
        pass
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

if __name__ == "__main__" :
    cfg = Config.fromfile(args.config)

    if args.quant :
        calib_files = glob.glob(os.path.join(args.calib_file_path, "*bin") )
        print("%d calib files for each model. " % (len(calib_files)//2) )
        np.random.shuffle(calib_files)
        rpn_calib_files = [x for x in calib_files if 'rpn' in x]
        pfe_calib_files = [x for x in calib_files if 'pfe' in x]
        

    ### for pfe engine creation 
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, \
    builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
    trt.OnnxParser(network,TRT_LOGGER) as parser, \
    builder.create_builder_config() as config:
        config.max_workspace_size = 1 << 30
        builder.max_batch_size = 1

        if args.quant:
            config.set_flag(trt.BuilderFlag.INT8)
            if args.minmax_calib:
                pfe_calib = MinMaxCalibrator(pfe_calib_files,  cache_file="pfe_calib_cache.bin", batch_size = args.calib_batch_size, shape = [cfg.max_pillars, cfg.max_points_in_voxel, cfg.feature_num] )
            else:
                pfe_calib = EntropyCalibrator(pfe_calib_files,  cache_file="pfe_calib_cache.bin", batch_size = args.calib_batch_size, shape = [cfg.max_pillars, cfg.max_points_in_voxel, cfg.feature_num] )
            config.int8_calibrator = pfe_calib
        else : 
            config.set_flag(trt.BuilderFlag.FP16)

        parsed = parser.parse_from_file(args.pfe_onnx_path)
        if parsed:
            print("building pfe trt engine . . .")
            serialized_engine = builder.build_serialized_network(network,config)
            with open(args.pfe_engine_path, 'wb') as f:
                f.write(serialized_engine)

            print("deserialize the engine . . . ")
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            context = engine.create_execution_context()
            print("context_pfe", context)

        else:
            print("Parsing Failed ! ")
            for i in range(parser.num_errors):
                print(parser.get_error(i))



    ### for rpn
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, \
    builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
    trt.OnnxParser(network,TRT_LOGGER) as parser, \
    builder.create_builder_config() as config:
        config.max_workspace_size = 1 << 30
        builder.max_batch_size = 1
        if args.quant:
            config.set_flag(trt.BuilderFlag.INT8)
            if args.minmax_calib:
                rpn_calib = MinMaxCalibrator(rpn_calib_files,  cache_file="rpn_calib_cache.bin", batch_size = args.calib_batch_size, shape = [cfg.pfe_output_dim, cfg.bev_h, cfg.bev_w] )
            else:
                rpn_calib = EntropyCalibrator(rpn_calib_files,  cache_file="rpn_calib_cache.bin", batch_size = args.calib_batch_size, shape = [cfg.pfe_output_dim, cfg.bev_h, cfg.bev_w] )
            config.int8_calibrator = rpn_calib
        else : 
            config.set_flag(trt.BuilderFlag.FP16)
            pass
        parsed = parser.parse_from_file(args.rpn_onnx_path)
        if parsed:
            print("building rpn trt engine . . .")
            serialized_engine = builder.build_serialized_network(network,config)
            with open(args.rpn_engine_path, 'wb') as f:
                f.write(serialized_engine)

            print("deserialize the engine . . . ")
            runtime = trt.Runtime(TRT_LOGGER)
            rpn_engine = runtime.deserialize_cuda_engine(serialized_engine)
            rpn_context = rpn_engine.create_execution_context()
            print("context_rpn", rpn_context)

        else:
            print("Parsing Failed ! ")
            for i in range(parser.num_errors):
                print(parser.get_error(i))


















