import sys
import copy
import json
import os
import sys
import numpy as np
import torch
from torch import nn
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config

from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2,pdb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="export a detector")
    parser.add_argument("--config", help="train config file path",type=str, default='waymo_centerpoint_pp_two_pfn_stride1_3x.py')
    parser.add_argument("--ckpt", help="ckpt of the model",type =  str )
    parser.add_argument("--pfe_save_path", help="the dir to save pfe  onnx",type = str, default = "pfe.onnx")
    parser.add_argument("--rpn_save_path", help="the dir to save rpn  onnx",type = str, default = "rpn.onnx")
    
    args = parser.parse_args()
    return args

args  = parse_args()




def example_to_device(example, device=None, non_blocking=False) -> dict:
    assert device is not None
    example_torch = {}
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points"]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v
    return example_torch

class PointPillars(nn.Module):
    def __init__(self,model):
        super(PointPillars, self).__init__()
        self.model = model
    def forward(self, x):
        x = self.model.neck(x)
        preds = self.model.bbox_head(x)
        for task in range(len(preds)):
            hm_preds = torch.sigmoid(preds[task]['hm'])
            preds[task]['dim'] = torch.exp(preds[task]['dim'])
            scores, labels = torch.max(hm_preds, dim=1)
            preds[task]["hm"] = (scores, labels)
        return preds

cfg = Config.fromfile(args.config)
cfg.EXPORT_ONNX = True
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
dataset = build_dataset(cfg.data.val)
data_loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=None,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_kitti,
    pin_memory=False,
)
checkpoint = load_checkpoint(model, args.ckpt,map_location="cpu")
model.eval()
model = model.cuda()
gpu_device = torch.device("cuda")
data_iter = iter(data_loader)
data_batch = next(data_iter)

pp_model = PointPillars(model)
with torch.no_grad():
    example = example_to_device(data_batch, gpu_device, non_blocking=False)
    example["voxels"] = torch.zeros((example["voxels"].shape[0],example["voxels"].shape[1],10),dtype=torch.float32,device=gpu_device)
    example.pop("points")
    example["shape"] = torch.tensor(example["shape"], dtype=torch.int32, device=gpu_device)
    pfe_inputs = torch.empty(args.max_pillars,20,10)
    pfe_inputs[:example["voxels"].shape[0]] = example['voxels']

    torch.onnx.export(model.reader,(pfe_inputs.cuda(),example["num_points"],example["coordinates"]), args.pfe_save_path,opset_version=11)

    rpn_input  = torch.randn((1,64,468,468),dtype=torch.float32,device=gpu_device)

    # getting errors with opset_version 11 , changed with 10
    torch.onnx.export(pp_model, (rpn_input), args.rpn_save_path ,opset_version=10)





points = data_batch['points'][:, 1:4].cpu().numpy()
MAX_PILLARS = 32000
save_path = "/mnt/data/waymo_opensets/val/calibrations/"
with torch.no_grad():
    for data_batch in data_loader:
        example = example_to_device(data_batch, gpu_device, non_blocking=False)
        token = example['metadata'][0]['token']
        voxels = example["voxels"]
        pfe_inputs = pad_voxel(voxels,example["num_points"],example["coordinates"],max_pillar_num=MAX_PILLARS)
        pfe_outputs = model.reader(pfe_inputs,None,None)
        torch_indexs = example['coordinates'][:,-1] + example['coordinates'][:,-2] * 468
        scatter_images = convert2scatter(pfe_outputs,torch_indexs).cpu().numpy()
        pfe_inputs = pfe_inputs.cpu().numpy()
        pfe_inputs.tofile(save_path + token[:-4] + "_pfe_input.npy")
        scatter_images.tofile(save_path + token[:-4] + "_rpn_input.npy")





