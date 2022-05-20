# -*- coding:utf-8 -*-

from __future__ import print_function
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu, boxes_iou3d_gpu
import shutil
from tqdm import tqdm
import os
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import sys
np.random.seed(0)


def read_pred(path):
    preds = np.loadtxt(path)
    # print(path, " has %d results..."%len(preds))
    return preds


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def theta_convert(src,dst):
    diff = np.abs(src - dst)
    a = diff if diff < 2*np.pi - diff else 2*np.pi - diff
    b = np.abs(diff - np.pi)
    if a > b:
        dst += np.pi
    return dst


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        bbox : (x,y,z,H,W,Z,theta, score)
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=13, dim_z=7)
        #
        self.kf.F = np.zeros((13, 13))
        self.kf.H = np.zeros((7, 13))

        # typecal linear formulation
        for i in range(7):
            self.kf.F[i, i] = 1
            if i < 6:
                self.kf.F[i, i+7] = 1
                self.kf.F[i+7, i+7] = 1
            self.kf.H[i, i] = 1
        
        self.kf.H[3:6, 3+7:6+7] = 0
        
        # self.kf.F[6,13] = 0.0  # for theta, no theta'
        # print(self.kf.F)

        self.kf.R[3:6, 3:6] *= 0  # 10.
        # self.kf.R[6, 6] = 0.01  # for theta, small R : believe in detection
        self.kf.R[6, 6] = 0.01     # for theta, big   R : believe in prediction

        self.kf.P[7:, 7:] *= 1000.  # give high unertainty to the unobservable initial velocities
        # self.kf.P *= 10.
        # self.kf.P *= 10

        # self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[7:, 7:] *= 0.0
        #  (x,y,z,H,W,Z,theta)
        self.kf.x[:7, 0] = bbox[:7]
        # 目前已经连续多少次未击中
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        # 目前总共击中了多少次
        self.hits = 0
        # 目前连续击中了多少次，当前step未更新则为0
        self.hit_streak = 0
        # 该tracker生命长度
        self.age = 0
        # add score and class
        self.score = 0
        self.label = 0

        # self.theta = 0    # zhanghao

        # print(self.kf.F)
        # print(self.kf.H)
        # print(self.kf.R)
        # print(self.kf.Q)
        # print(self.kf.P)

    def update(self, bbox):
        """
        bbox : [x,y,z,H,W,Z,theta,score,...]
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # self.kf.update(convert_bbox_to_z(bbox))
        self.kf.update(bbox[:7].reshape(-1, 1))
        # self.theta = bbox[6]  # zhanghao
        self.score = bbox[7]
        self.label = bbox[8]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        [x,y,z,H,W,Z,theta,score,...]
        """
        # if((self.kf.x[6]+self.kf.x[2])<=0):
        #   self.kf.x[6] *= 0.0
        # set min value of  H,W,Z, to zero.
        for i in range(3, 6):
            if (self.kf.x[i] + self.kf.x[i+7] <= 0):
                self.kf.x[i] *= 0.0

        self.kf.predict()


        self.age += 1
        if(self.time_since_update > 0):
        # if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # self.history.append(convert_x_to_bbox(self.kf.x))
        self.history.append(self.kf.x[:7].reshape(1, -1))
        if len(self.history) > 1:
            # print("shape : ", self.history[-1].shape, self.history[-2][0,6] )
            self.history[-1][0,6] = theta_convert(self.history[-2][0,6], self.history[-1][0,6])
            self.history[-1][0,2:6] = 0.9 * self.history[-2][0,2:6] + 0.1 * self.history[-1][0,2:6]

        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        # return convert_x_to_bbox(self.kf.x)
        # self.kf.x[6] = self.theta
        ret = np.append(self.kf.x[:7], self.score)
        ret = np.append(ret, self.label)
        
        return ret.reshape(1, -1)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        # TODO : why np.empty((0,5)) ?
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # N x M,直接调用pcdet底层iou算法
    # iou_matrix = iou_batch3d(detections, trackers)
    iou_matrix = boxes_bev_iou_cpu(detections[:, :7], trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # 每个detection最多只配到了一个tracker，或者每个tracker最多只配到了一个detection
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # matched_indices :  (N x 2)
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    # 在分配的基础上必须大于iou阈值才能算配对
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 9)), velocities=None   ):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,z,dx,dy,dz,r,score,class],[x,y,z,dx,dy,dz,r,score,class],...]

        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            # 找到tracker.predict ,   pos : [x,y,z,H,W,Z,theta]
            pos = self.trackers[t].predict()[0]
            # print("predict: ", pos)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # (N x 7)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 匈牙利算法 做协同
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :9 ])
        track2det = {m[1]:m[0] for m in matched}
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)

        # for trk in reversed(self.trackers):
        for trackidx in np.arange(len(self.trackers)-1, -1, -1):
            trk = self.trackers[trackidx]
            d = trk.get_state()[0]
            
            # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
            #     # ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            #     ret.append(np.append(d, trk.id+1))
            # i -= 1
            # # remove dead tracklet
            # if(trk.time_since_update > self.max_age):
            #     self.trackers.pop(i)

            # ## test
            if (trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
            # if trk.time_since_update and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                
                # ret.append(np.append(d, trk.id+1))
                # ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                if trackidx in track2det:
                    if  velocities is not None :
                        d = np.append(d,np.append(velocities[track2det[trackidx]], trk.id+1 ) )
                        ret.append( d)
                    else:
                        d = np.append(d,np.append(np.zeros(2), trk.id+1 ) )
                        ret.append(np.append(d, trk.id+1))

            i -= 1
            # remove dead tracklet
            if(trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
            # ## end 


        # （N,10）: [x,y,z,dx,dy,dz,r,score,class,track_id]
        if(len(ret) > 0):
            # return np.concatenate(ret)
            return np.stack(ret)
        return np.empty((0, 10))





