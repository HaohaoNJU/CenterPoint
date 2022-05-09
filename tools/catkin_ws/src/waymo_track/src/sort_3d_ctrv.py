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
from ukf import UKF
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


def kitti2waymo(bbox):
    bbox[:,6] = -(bbox[:,6] + np.pi /2 )
    return bbox[:, [0,1,2,4,3,5, 6]]

def theta_convert(src,dst):
    diff = np.abs(src - dst)
    a = diff if diff < 2*np.pi - diff else 2*np.pi - diff
    b = np.abs(diff - np.pi)
    if a > b:
        dst += np.pi
    return dst

def vote_decision(x):
    assert  len(x) > 0, "Shouldn't give an empty list or array"
    return max(set(x), key=x.count)

class ExpAvg(object):
    def __init__(self, value = [0], alpha = [0.9], cond = [None]):
        """
        params: alpha, exp average paramter.
        params: cond, conditional function returns bool value, whether or not to skip out of the smooth loop. 
        """
        assert len(alpha) == len(cond), 'alphas should have the same elements as conditions ! '
        self.alpha = alpha
        self.cond = cond
        self.x = value
        self.dim = len(alpha)

    def set(self, value):
        assert self.dim == len(value), 'Given values should have the same elements as conditions ! '
        for i, v in enumerate(value):
            if self.cond[i] is None or self.cond[i](self.x[i], v):
                self.x[i] = v
            else:
                self.x[i] = self.alpha[i] * self.x[i] + (1-self.alpha[i]) * v

    def get_state(self):
        return self.x


# For linear model
def linear_process_model(x, dt, rand_var):
    """
    x : state_vector [  px. py, vx, vy ]
    dt : delta time from last state timestamp, float by second 
    rand_var : [nu_a, nu_dot_psi], gaussions of acceleration and angle acceleration
   
    y
    ^
     |
     |
     |
    (.)-------------------> x
    z    Along z-axis anti-clockwise is the yaw rotation. 
    """
    assert len(x) == 4 and len(rand_var) == 2, "We need 4 dim state vectors and  2 randoms, nu_v and nu_psi_dot !"
    (px, py, vx,vy) = x  
    nu_ax = rand_var[0] ; nu_ay = rand_var[1] 
    tayler1 = np.zeros_like(x)
    tayler2 =  np.zeros_like(x)


    tayler1[0] = vx * dt
    tayler1[1] = vy * dt
    tayler1[2] = nu_ax* dt 
    tayler1[3] = nu_ay * dt
    # pre-estimated terms , assuming dpsi=0, ddpsi=0, nu_ddpsi=0
    tayler2[0] = dt**2  *  nu_ax / 2
    tayler2[1] = dt**2  *  nu_ay / 2
    return  x + tayler1 + tayler2
    
    # tayler1[0] = nu_ax * dt
    # tayler1[1] = nu_ay * dt
    # return x + tayler1

class UKFBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, velocity = None,  smoother = None, dt = 0.1, max_labels = 20, smooth_value = None, label=None):
        """
        Initialises a tracker using initial bounding box.
        """
        # bbox[2] *= -1
        self.dt = dt
        std_laspx = 0.1 ; std_laspy = 0.1 ; std_laspsi = 1.0; std_lasv = 1.0; 
        self.measurement_noise_covar =  np.array([
            [std_laspx**2, 0, 0, 0],
            [0, std_laspy**2, 0, 0],
            [0, 0, std_laspsi**2, 0],
            [0, 0, 0, std_lasv**2],]) * 100

        P = np.array([
            [1, 0,0,0,0],
            [0,1,0,0,0],
            [0,0,100,0,0],
            [0,0,0,100,0],
            [0,0,0,0,1000]
        ],dtype=np.float32)



        ## For CTRV process model . 
        initial_state = np.zeros(5,dtype=np.float32)
        initial_state[:2] = bbox[:2]
        if velocity is not None:
            velo_head = np.arctan2(velocity[1], velocity[0])
            linear_velo = np.sqrt(np.sum(  velocity**2)  )
            initial_state[3] = linear_velo
            initial_state[2] = velo_head
        # state_vector [  px. py, psi, v, dot_psi ]
        # ukf = UKF(initial_state=initial_state,initial_covar=P * 0.001,iterate_function = ctrv_process_model)
        # self.ukf = UKF(initial_state = initial_state, initial_covar = P * 0.001,std_ddpsi = 1, std_a = 3)


        # P = np.array([
        #     [1, 0,0,0],
        #     [0,1,0,0],
        #     [0,0,100,0],
        #     [0,0,0,100],
        # ],dtype=np.float32)

        Q = np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,10,0,0],
            [0,0,0,10,0],
            [0,0,0,0,100]
        ],dtype=np.float32) * 10
        initial_state = np.zeros(4)
        # initial_state[:2] = bbox[:2]
        # if velocity is not None:
        #     initial_state[2:] = velocity[:2] 

        # self.ukf = UKF(num_states = 4, initial_state=initial_state,initial_covar=P * 0.001,iterate_function = linear_process_model, std_ddpsi = 1, std_a = 3)

        # params when vx,vy is noise 
        self.ukf = UKF(num_states = 5, 
                       initial_state=initial_state,
                       initial_covar=P * 1.0,
                       q_matrix = Q,
                       std_ddpsi = 100, 
                       std_a = 100)


        self.id = UKFBoxTracker.count
        UKFBoxTracker.count += 1
        self.history = {'labels':[label], "states":[self.ukf.get_state()], "smooth_values":[smooth_value]}

        # 目前总共击中了多少次, totally how many times has been updated with measurement !
        self.hits = 0
        # 目前连续击中了多少次，当前step未更新则为0
        self.hit_streak = 0
        # 目前已经连续多少次未击中
        self.time_since_update = 0
        # 该tracker生命长度(Totally how many times has been predicted ! )
        self.age = 0
        # For label vote
        self.labels = []
        self.max_labels = max_labels
        # For smooth values 
        self.smoother = smoother
        self.yaw = bbox[2]

    def update(self, states = None, label = None, smooth_value = None, velocity=None):
        """
        bbox : [x,y,z,H,W,Z,theta,score,...]
        Updates the state vector with observed bbox.
        """
        # states[2] *= -1
        self.yaw = states[2]
        states = states[:2]

        self.time_since_update = 0
        # self.history = []
        self.hits += 1
        self.hit_streak += 1

        r_matrix = self.measurement_noise_covar[:2,:2] if velocity is None else self.measurement_noise_covar
        state_idx = [0,1] if velocity is None else [0,1,2,3]

        ### For CTRV Model         
        if velocity is not None:
            velo_head = np.arctan2(velocity[1], velocity[0])
            linear_velo = np.sqrt(np.sum(  velocity**2)  )
            states  = np.append(states, velo_head)
            states  = np.append(states, linear_velo)

        ### For linear model 
        # if velocity is not None:
        #     states  = np.append(states, velocity[:2])

        # states =  states if velocity is None else np.append(states,velocity)
        self.ukf.update(state_idx = state_idx, data = states, r_matrix = r_matrix)
        # self.kf.update(bbox[:7].reshape(-1, 1))
        # self.theta = bbox[6]  # zhanghao
        # self.score = bbox[7]
        # smooth the values 
        if smooth_value is not None:
            if not self.smoother:
                print("Warning : smoother is not available !")
            else:
                self.smoother.set(smooth_value)
        if label is not None:
            self.labels.append(label)
            self.labels = self.labels[-self.max_labels:]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        [x,y,z,H,W,Z,theta,score,...]
        """

        self.ukf.predict(self.dt)
        self.age += 1
        if(self.time_since_update > 0):
        # if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # self.history.append(convert_x_to_bbox(self.kf.x))
        # self.history.append(self.kf.x[:7].reshape(1, -1))
        state = np.array(self.ukf.get_state() ) 
        # state[2] *= -1
        self.history["states"].append( state)
        if self.smoother:
            self.history['smooth_values'].append( np.array(self.smoother.get_state()))
        if len(self.labels):
            self.history['labels'].append(self.labels[-1])


        # if len(self.history) > 1:
        #     # print("shape : ", self.history[-1].shape, self.history[-2][0,6] )
        #     self.history[-1][0,6] = theta_convert(self.history[-2][0,6], self.history[-1][0,6])
        #     self.history[-1][0,2:6] = 0.9 * self.history[-2][0,2:6] + 0.1 * self.history[-1][0,2:6]

        # return self.history[-1]
        return self.get_state()

    def get_state(self):
 
        return {k[:-1]:v[-1] for k,v in self.history.items()}


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # N x M,直接调用pcdet底层iou算法
    # iou_matrix = iou_batch3d(detections, trackers)
    # kitti_dets = kitti2waymo(detections)
    # kitti_trks = kitti2waymo(trackers)
    iou_matrix = boxes_bev_iou_cpu(detections[:, :7], trackers)
    # iou_matrix = boxes_bev_iou_cpu(kitti_dets, kitti_trks)
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

    def update(self, dets=np.empty((0, 9)), velocities = None ):
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
            # 找到tracker.predict ,   pos : {'state':[cx,cy,psi,v,dpsi],'label':label,"smooth_value":[z,dx,dy,dz,score]}
            pos = self.trackers[t].predict()
            state = pos['state']
            # replace   psi with yaw angle 
            state[2] = self.trackers[t].yaw
            smooth = pos['smooth_value']
            # trt : [cx,cy,cz,dx,dy,dz]
            # trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            # trt[:] = pos['state'][:2] + pos['smooth_value'][:4] + pos['state'][2:3]
            trk[:] =[state[0],state[1],smooth[0],smooth[1],smooth[2],smooth[3],state[2]]
             
            # if np.any(np.isnan(pos)):
            #     to_del.append(t)

        # (N x 7)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 匈牙利算法 做协同
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            # update( states, label = None, smooth_value = None, velocity=None):
            # self.trackers[m[1]].update(dets[m[0], :])
            meas = dets[m[0],[0,1,6]]
            v = None if velocities is None else velocities[m[0]]
            self.trackers[m[1]].update( states = meas, 
                                        label = dets[m[0],-1], 
                                        smooth_value = dets[m[0], [2,3,4,5,-2]],
                                        velocity = v)
        
        track2det = {m[1]:m[0] for m in matched}

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # bbox, velocity = None,  smoother = None
            v = None if velocities is None else velocities[i]

            # alpha = [0.9], cond = [None]
            cond = [lambda a,b : np.abs(a-b)/min(np.abs(a),np.abs(b)) > 0.2,
                    lambda a,b : np.abs(a-b)/min(np.abs(a),np.abs(b)) > 0.2,
                    lambda a,b : np.abs(a-b)/min(np.abs(a),np.abs(b)) > 0.2,
                    lambda a,b : np.abs(a-b)/min(np.abs(a),np.abs(b)) > 0.2,
                    None]
            alpha = [0.9,0.9,0.9,0.9,0.5]
            trk = UKFBoxTracker(bbox = dets[i,[0,1,6]],
                                velocity = v,
                                smoother = ExpAvg(value = dets[i,[2,3,4,5,-2]] , alpha = alpha, cond = cond),   
                                smooth_value = dets[i,[2,3,4,5,-2]],
                                label = dets[i,-1]                            
                                )
            self.trackers.append(trk)

        i = len(self.trackers)

        # for trk in reversed(self.trackers[: len(self.trackers)-len(unmatched_dets) ]):
        for track_idx, det_idx in track2det.items():

            trk = self.trackers[track_idx]

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
                # ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                # ret.append(np.append(d, trk.id+1))
                trk_dict = trk.get_state() 
                state = trk_dict['state']
                # replace psi with yaw angle 
                # state[2] = trk.yaw
                smooth = trk_dict['smooth_value']
                label = trk_dict['label']
                # trt : [cx,cy,cz,dx,dy,dz]
                # trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
                # trt[:] = pos['state'][:2] + pos['smooth_value'][:4] + pos['state'][2:3]
                # 11 dof (x,y,z,dx,dy,dz,yaw,score, label, v, id)
                d =[state[0],state[1],smooth[0],smooth[1],smooth[2],smooth[3],trk.yaw, smooth[4], label,  state[2], state[3],trk.id+1]
                ret.append(d)
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









