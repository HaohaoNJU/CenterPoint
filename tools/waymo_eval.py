
"Do inference and evaluation offline"
from random import sample
import sys
from numpy.core.fromnumeric import argsort
import json
import os,shutil, sys
sys.path.append('..')
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp

import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pickle as pkl
from multiprocessing import Pool
#For Metics Computation
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2
import tensorflow as tf

def kitti2waymo(bbox):
    bbox[:,6] = -(bbox[:,6] + np.pi /2 )
    return bbox[:, [0,1,2,4,3,5, 6]]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path",type=str, default='../configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py')
    parser.add_argument("--ckpt", help="ckpt of the model",type =  str,  default = "../waymo_centerpoint_ckpts/waymo_centerpoint_pp_two_pfn_stride1_3x/epoch_36.pth")
    parser.add_argument("--save_path", help="the dir to save outputs",type = str, default = "./save_dir_tmp")
    parser.add_argument("--info_path", help="the path to gt infos",type = str, default = "/mnt/data/waymo_opensets/infos_val_01sweeps_filter_zero_gt.pkl")
    parser.add_argument("--num_worker",type=int, default=1, help="num workers to infers")
    parser.add_argument("--score_thre",type=float, default=0.2, help="as is named, the score threshold")

    parser.add_argument("--cpp_output",action = "store_true", default=False, help="num workers to infers")
    parser.add_argument("--track_output",action = "store_true", default=False, help="num workers to infers")

    parser.add_argument("--run_infer", action="store_true",default=False)
    parser.add_argument("--subset",action="store_true",default=False)
    parser.add_argument("--batch_size", type=int, default = 1, help="batch size of samples ")
    args = parser.parse_args()
    return args


CLASSNAME2LABEL = {"VEHICLE" : 0, "PEDESTRIAN" : 1, "CYCLIST":2,"SIGN":3, "UNKNOWN":4}
# LABEL_MAP = {0:1, 1:2, 2:0, 3:3, 4:4}
LABEL_MAP = {0:1, 1:2, 2:4, 3:0, 4:0}

PRED_LABEL = {"VEHICLE" : 0, "PEDESTRIAN" : 1, "CYCLIST":2 }
ANNO_LABEL = {'UNKNOWN':0, 'VEHICLE':1, 'PEDESTRIAN':2, 'SIGN':3, 'CYCLIST':4}

BATCH_SIZE = 1
config_text = None
class DetectionMetricsEstimatorTest(tf.test.TestCase):
    def _BuildConfig(self):
        config = metrics_pb2.Config()
        global config_text
        config_text = """
        num_desired_score_cutoffs: 11
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.5
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_2D
        """
        text_format.Merge(config_text, config)
        return config
    def _BuildGraph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
      
            metrics = detection_metrics.get_detection_metric_ops(
                config=self._BuildConfig(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=tf.zeros_like(
                    self._pd_frame_id, dtype=tf.bool),
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=tf.ones_like(
                    self._gt_frame_id, dtype=tf.uint8),
                recall_at_precision=0.95,
            )
        return metrics

    def _EvalUpdateOps(
      self,
      sess,
      graph,
      metrics,
      prediction_frame_id,
      prediction_bbox,
      prediction_type,
      prediction_score,
      ground_truth_frame_id,
      ground_truth_bbox,
      ground_truth_type,
      ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
            })

    def _EvalValueOps(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

def example_to_device(example, device=None, non_blocking=False) -> dict:
    assert device is not None
    example_torch = {}
    float_names = ["voxels", "bev_map"]
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

def pkl_read(p):
	data = pkl.load(open(p,'rb'))
	return data
def load_anno_gt(anno_path):
    all_data = pkl_read(anno_path)

    gt_boxes = np.array([ann['box'] for ann in all_data['objects']]).reshape(-1, 9)
    gt_classes = np.array([ann['label'] for ann in all_data['objects']])
    return gt_boxes, gt_classes

def classname2label(data):
    # return np.array([CLASSNAME2LABEL[x] for x in data])
    return np.array([ANNO_LABEL[x] for x in data])

def dict_to_cpu(data):
    for k,v in data.items():
        if hasattr(v,"cpu"):
            data[k] = v.cpu().numpy()
    return data
def save_predictions(worker_id, data_loader,model,gpu_device, args):
    with torch.no_grad() :
        for idx, data_batch in tqdm(enumerate(data_loader)):
        # for idx in tqdm(range(data_loader.__len_())):
            if  idx % args.num_worker != worker_id:
                continue
            example = example_to_device(data_batch,device=gpu_device, non_blocking=True)

            rets = model(example,return_loss=False)
            for i, ret in enumerate(rets):
                ret = rets[i]
                ret = dict_to_cpu(ret)
                token = ret['metadata']['token']
                sp = os.path.join(args.save_path,token)
                pkl.dump(ret, open(sp,'wb'))

def run_inference(args):
    cfg = Config.fromfile(args.config)
    if args.subset :
        cfg.data.val.root_path="/mnt/data/waymo_opensets/val_sub0.1"
        pattern = "infos_val_02sweeps_filter_zero_gt.pkl" if  "two_sweep" in args.config else "infos_val_01sweeps_filter_zero_gt.pkl"
        cfg.data.val.info_path= os.path.join(cfg.data.val.root_path, pattern)
        cfg.data.val.ann_file= os.path.join(cfg.data.val.root_path, pattern)
    else:
        cfg.data.val.root_path="/mnt/data/waymo_opensets"
        pattern = "infos_val_02sweeps_filter_zero_gt.pkl" if  "two_sweep" in args.config else "infos_val_01sweeps_filter_zero_gt.pkl"
        cfg.data.val.info_path= os.path.join(cfg.data.val.root_path, pattern)
        cfg.data.val.ann_file= os.path.join(cfg.data.val.root_path, pattern)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)


    dataset = build_dataset(cfg.data.val)
    data_loader = DataLoader(
        dataset,batch_size=BATCH_SIZE,sampler=None,shuffle=False,num_workers=0,collate_fn=collate_kitti,pin_memory=False,
    )
    # data_iter = iter(data_loader)
    # data_batch = next(data_iter)
    ckpt_path = cfg.PRETRAINED if args.ckpt is None else args.ckpt
    checkpoint = load_checkpoint(model, ckpt_path,map_location="cpu")
    print("Model has been loaded from %s ."  %  ckpt_path)

    gpu_device = torch.device("cuda")

    print("Running  Inference . . .")

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            if torch.any(torch.isnan(m.running_mean)):
                m.running_mean.zero_()
                print("running mean  set to 0 ")
            if torch.any(torch.isnan(m.running_var)):
                m.running_var.fill_(1)
                print("running var set to  1")

    model.eval()
    model.cuda()

    # if args.num_worker == 1:
    save_predictions(0,data_loader,model,gpu_device, args)

    # else:
        # mp.spawn(save_predictions, nprocs = args.num_worker, args = (data_loader, model,gpu_device,  args))
        # pool = Pool(args.num_worker)
        # try:
        #     for worker_id in range(args.num_worker):
        #         pool.apply_async(save_predictions, (worker_id,data_loader,model,args))
        #     pool.close()
        #     pool.join()
        # except Exception as e:
        #     print(e)
        # finally:
        #     print("Success finished saveing detection outputs !")


def compute_detection_metrics(args):
    SCORE_THRE = args.score_thre
    
    val_infos = pkl_read(args.info_path)
    sample_detection_metrics = DetectionMetricsEstimatorTest()
    graph = tf.Graph()
    metrics = sample_detection_metrics._BuildGraph(graph)
    cnt = 0
    # with sample_detection_metrics.test_session(graph=graph) as sess:
    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.initializers.local_variables())
        pred_boxes , pred_classes , pred_scores, pred_frame_ids,gt_frame_ids, gt_boxes, gt_classes =[], [],[],[],[],[],[]
        for idx, info in tqdm(enumerate(val_infos)):
            if args.cpp_output :
                pred_name = os.path.basename(info['path']).replace("pkl","bin.txt")
            elif args.track_output:
                pred_name = os.path.basename(info['path']).replace("pkl","txt")
            else:
                pred_name = os.path.basename(info['path'])

            pred_file = os.path.join(args.save_path,pred_name)

            if not os.path.exists(pred_file):
                continue
            if args.cpp_output:
                data = np.loadtxt(pred_file) 
                if not len(data) or len(data.shape)==1 :    continue
                # cx,cy,cz,dx,dy,dz,vx,vy,rot,score, lab
                scores = data[:,-2]
                box3d_lidar = data[:,[0,1,2,3,4,5,8]]
                label_preds = data[:,10].astype(np.int8)
            elif args.track_output:
                data = np.loadtxt(pred_file)
                if not len(data) or len(data.shape)==1 :    continue
                scores = data[:,-3]
                box3d_lidar = data[:,[2,3,4,5,6,7,8]]
                box3d_lidar = kitti2waymo(box3d_lidar)
                label_preds = data[:,1]
            else:
                data = pkl_read(pred_file)
                scores = data['scores']
                box3d_lidar = data['box3d_lidar'][:,[0,1,2,3,4,5,-1]]
                label_preds = data['label_preds']
            # box3d_lidar[:,[3,4]] = box3d_lidar[:,[4,3]]
            # box3d_lidar[:,-1] = -box3d_lidar[:,-1]

            selected_indexs = np.where(scores >= SCORE_THRE)[0]

            if len(selected_indexs)==0:
                continue
            # print("pred_boxes : \n",scores )
            # print("gt_boxes : \n",classname2label(info['gt_names']) )
            cnt += 1

            ############################################################
            gt_boxes.append(info['gt_boxes'][:,[0,1,2,3,4,5,8]])
            gt_classes.append(classname2label(info['gt_names']))
            gt_frame_ids.append(np.ones(len(info['gt_boxes'])) * idx)
            
            # box3d_lidar = kitti2waymo(box3d_lidar)
            label_preds = np.array([LABEL_MAP[int(i)] for i in label_preds])
            ############################################################
            pred_boxes.append(box3d_lidar[selected_indexs])
            pred_classes.append(label_preds[selected_indexs])
            pred_scores.append(scores[selected_indexs])
            pred_frame_ids.append(np.ones(len(selected_indexs)) * idx)
            if idx % 50 ==0 or idx == len(val_infos)-1:
                pred_boxes = np.concatenate(pred_boxes,axis=0)
                pred_classes = np.concatenate(pred_classes,axis = 0)
                pred_scores = np.concatenate(pred_scores,axis=0)
                pred_frame_ids = np.concatenate(pred_frame_ids)
                gt_boxes = np.concatenate(gt_boxes)
                gt_classes = np.concatenate(gt_classes)
                gt_frame_ids = np.concatenate(gt_frame_ids)

                sample_detection_metrics._EvalUpdateOps(sess, graph, metrics, pred_frame_ids, pred_boxes, pred_classes,\
                            pred_scores, gt_frame_ids, gt_boxes, gt_classes)
                pred_boxes , pred_classes , pred_scores, pred_frame_ids, gt_frame_ids, gt_boxes, gt_classes = [],[],[],[],[],[],[]
          # Looking up an exisitng var to check that data is accumulated properly
          # in the variable
        aps = sample_detection_metrics._EvalValueOps(sess, graph, metrics)
        print("EVALUATION RESULTS OF {} SAMPLES FROM SCORE_THRE {}\n".format(cnt,SCORE_THRE))

        for k,v in aps.items():
            print(k, v)
        return aps



if __name__ == "__main__":
    args = parse_args()
    assert not (args.cpp_output and args.track_output ), "cpp_output confilect with track_output ! "
    if args.subset:
        args.info_path = "/mnt/data/waymo_opensets/val_sub0.1/infos_val_01sweeps_filter_zero_gt.pkl"
    os.makedirs(args.save_path,exist_ok=True)
    # global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    if args.run_infer:
        run_inference(args)

    print("Computing 2D Detection Metrics . . .")
    compute_detection_metrics(args)
    print("Metrics : \n", config_text)



        




















