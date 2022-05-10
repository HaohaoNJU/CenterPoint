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
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="export a detector")
    parser.add_argument("--config", help="train config file path",type=str, default='waymo_centerpoint_pp_two_pfn_stride1_3x.py')
    parser.add_argument("--ckpt", help="ckpt of the model",type =  str )    
    parser.add_argument("--calib_file_path", help="the dir to calibration files, only config when `quant` is enabled. ",type = str)
    args = parser.parse_args()
    return args

args  = parse_args()
cfg = Config.fromfile(args.config)
cfg.EXPORT_ONNX = True

def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

def pad_voxel(features, num_voxels, coors, max_pillar_num = None):
    vx = cfg.x_step
    vy = cfg.y_step
    x_offset = vx/2 - cfg.x_range
    y_offset = vy/2 - cfg.y_range
    _with_distance = False
    
    device = features.device
    dtype = features.dtype
    # Find distance of x, y, and z from cluster center
    # features = features[:, :, :self.num_input]
    points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
        features
    ).view(-1, 1, 1)
    f_cluster = features[:, :, :3] - points_mean
    # Find distance of x, y, and z from pillar center
    # f_center = features[:, :, :2]
    f_center = torch.zeros_like(features[:, :, :2])
    f_center[:, :, 0] = features[:, :, 0] - (
        coors[:, 3].to(dtype).unsqueeze(1) * vx + x_offset
    )
    f_center[:, :, 1] = features[:, :, 1] - (
        coors[:, 2].to(dtype).unsqueeze(1) * vy + y_offset
    )
    
    # Combine together feature decorations
    features_ls = [features, f_cluster, f_center]
    if _with_distance:
        points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
        features_ls.append(points_dist)
    features = torch.cat(features_ls, dim=-1)
    # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
    # empty pillars remain set to zeros.
    voxel_count = features.shape[1]
    mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask
    if max_pillar_num is not None:
        pillar_size  = [x for x in features.shape]
        if max_pillar_num < pillar_size[0]:
            features = features[:max_pillar_num]
        else:
            pillar_size[0] = max_pillar_num - pillar_size[0]
            zeros = torch.zeros(pillar_size).to(features.device)
            features = torch.cat([features, zeros],axis = 0)
    return features


def convert2scatter(inputs,indexs):
    assert len(inputs.shape) == 2
#     assert len(inputs) == len(indexs)
    dim = inputs.shape[-1]
    rets = torch.zeros((dim,cfg.bev_h,cfg.bev_w),dtype=inputs.dtype).to(inputs.device)
    num_pillars = min(len(indexs), cfg.max_pillars)
    for i in range(num_pillars):
        if indexs[i] <0 or indexs[i] >= cfg.bev_w * cfg.bev_h: continue
        yIdx = indexs[i] // cfg.bev_w
        xIdx = indexs[i] % cfg.bev_h
        rets[:,yIdx,xIdx] = inputs[i]
    return rets


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


model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
dataset = build_dataset(cfg.data.val)
data_loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=None,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_kitti,
    pin_memory=False,
)
checkpoint = load_checkpoint(model, args.ckpt,map_location="cpu")
model.eval()
model = model.cuda()
gpu_device = torch.device("cuda")

with torch.no_grad():
    for batch_idx, data_batch in tqdm(enumerate(data_loader)):
        if batch_idx > 1000:
            break
        example = example_to_device(data_batch, gpu_device, non_blocking=False)
        token = example['metadata'][0]['token']
        voxels = example["voxels"]
        pfe_inputs = pad_voxel(voxels,example["num_points"],example["coordinates"],max_pillar_num=cfg.max_pillars)
        pfe_outputs = model.reader(pfe_inputs,None,None)
        torch_indexs = example['coordinates'][:,-1] + example['coordinates'][:,-2] * cfg.bev_h
        scatter_images = convert2scatter(pfe_outputs,torch_indexs).cpu().numpy()
        pfe_inputs = pfe_inputs.cpu().numpy()
        pfe_inputs.tofile(os.path.join(args.calib_file_path , token[:-4] + "_pfe_input.bin"))
        scatter_images.tofile( os.path.join(args.calib_file_path ,  token[:-4] + "_rpn_input.bin"))





