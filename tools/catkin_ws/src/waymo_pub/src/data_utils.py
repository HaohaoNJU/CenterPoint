#!/usr/bin/python3

import os
import cv2
import tqdm
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm
matplotlib.use("cairo")
import tensorflow as tf
# tf.enable_eager_execution()

tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import transform


cmap = matplotlib.cm.get_cmap("viridis")


def get_one_frame_tfrecord(tfrecord_path, sensorID=0):    
    # 这个方法非常慢，不建议使用，建议先将图片、点云保存到磁盘上(参考 det3d 做法)再操作
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        image = tf.image.decode_jpeg(frame.images[sensorID].image)

        range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                                        frame,
                                        range_images,
                                        camera_projections,
                                        range_image_top_pose)

        yield np.array(image)[...,::-1], np.concatenate(points, axis=0)

        # cv2.imshow("image", np.array(image)[...,::-1])
        # cv2.waitKey(1)


def get_lidar_pkl(pkl_path):
    # 加载 det3d 转成的 pkl
    data = pickle.load(open(pkl_path, 'rb'))
    xyz = data['lidars']['points_xyz']
    intensity = data['lidars']['points_feature'][:,0].reshape(-1,1)  # [:,0]: 反射率 [:,1]: 伸长率
    
    return np.hstack((xyz, intensity))
    # return xyz


def get_box3d_pkl(pkl_path, with_tra=True):
    # print(pkl_path)
    data = pickle.load(open(pkl_path, 'rb'))
    boxes9d = np.concatenate([obj['box'] for obj in data['objects']] ,axis=0).reshape(-1,9)
    boxes9d[:, 8] += np.pi / 2
    # boxes7d = np.hstack((boxes9d[:, :6], boxes9d[:, 8].reshape(-1,1)))
    boxes7d = np.hstack((boxes9d[:, [0,1,2,4,3,5]], -boxes9d[:, 8].reshape(-1,1)))

    labels = np.array([obj['label'] for obj in data['objects']])
    # ids = [obj['id'] for obj in data['objects']]  # 这个 id 只是序号
    ids = np.array([hash(obj['name'])%1999 for obj in data['objects']])
    # global_speed = [obj['global_speed'] for obj in data['objects']]

    
    filter_value = 100 if with_tra else 3
    mask = labels < filter_value 
    labels = labels[mask]
    LABEL_MAP = {0:3, 1:0, 2:1, 3:2}  # {0:"Cyc", 1:"Car", 2:"Ped", 3:"Tra"} if GT else {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}
    labels_remap = np.array([LABEL_MAP[i] for i in labels])

    return boxes7d[mask], labels_remap, ids[mask]

    # mask2 = boxes7d[:, 1] < 15  # -5
    # mask3 = boxes7d[:, 1] > 0  # -15
    # mask4 = boxes7d[:, 0] > 0  # 25
    # mask5 = boxes7d[:, 0] < 10  # 26
    # maskAll = mask2 * mask3 * mask4 * mask5
    # return np.array(boxes7d[maskAll]), np.array(labels)[maskAll], np.array(ids)[maskAll]


def get_cv_image(img_path):
    img = cv2.imread(img_path)
    return np.empty(0) if img is None else img


def draw_bbox(img, bboxes):
    for box in bboxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    return img
    

def serialize_image_tfrecord(tfrecord_path, save_path, sensorID=0, prefix="seq_0_frame_"):
    # 仅保存一个 cam 的图片，且不画任何 box
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    frame = open_dataset.Frame()

    for idx, data in tqdm.tqdm(enumerate(dataset)):
        frame.ParseFromString(bytearray(data.numpy()))
        image = tf.image.decode_jpeg(frame.images[sensorID].image)
        cv_image = np.array(image)[...,::-1]
        cv2.imwrite(os.path.join(save_path + prefix+"%d.jpg"%idx), cv_image)


def serialize_5image_tfrecord(tfrecord_path, save_path="", prefix="seq_0_frame_"):
    print(tfrecord_path)

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    for ii, data in tqdm.tqdm(enumerate(dataset)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        plt.figure(figsize=(15, 10))
        
        for index, image in enumerate(frame.images):
            # cv_img = np.array(tf.image.decode_jpeg(image.image))[...,::-1]
            # cv2.imshow("a", cv_img)
            # cv2.waitKey(0)
            ax = plt.subplot(2, 3, index+1)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            for camera_label in frame.camera_labels:
                # Ignore camera labels that do not correspond to this camera.
                if camera_label.name != image.name:
                    continue
                
                for label in camera_label.labels:
                    # Draw the object bounding box.
                    ax.add_patch(patches.Rectangle(
                        xy=(label.box.center_x - 0.5 * label.box.length,
                            label.box.center_y - 0.5 * label.box.width),
                            width=label.box.length,
                            height=label.box.width,
                            linewidth=1,
                            edgecolor='yellow',
                            facecolor='none'))

            plt.imshow(tf.image.decode_jpeg(image.image), cmap=None)
            plt.title(open_dataset.CameraName.Name.Name(image.name))
            plt.grid(False)
            plt.axis('off')
            plt.savefig(os.path.join(save_path + prefix+"%d.jpg"%ii))
            # plt.clf()


def serialize_5image_frame(frame, save_path=""):
    plt.figure(figsize=(15, 10))
    
    for index, image in enumerate(frame.images):
        ax = plt.subplot(2, 3, index+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        for camera_label in frame.camera_labels:
            # Ignore camera labels that do not correspond to this camera.
            if camera_label.name != image.name:
                continue
            
            for label in camera_label.labels:
                # Draw the object bounding box.
                ax.add_patch(patches.Rectangle(
                    xy=(label.box.center_x - 0.5 * label.box.length,
                        label.box.center_y - 0.5 * label.box.width),
                        width=label.box.length,
                        height=label.box.width,
                        linewidth=1,
                        edgecolor='yellow',
                        facecolor='none'))

        plt.imshow(tf.image.decode_jpeg(image.image), cmap=None)
        plt.title(open_dataset.CameraName.Name.Name(image.name))
        plt.grid(False)
        plt.axis('off')
        plt.savefig(save_path)


def compute_3d_cornors(x, y, z, dx, dy, dz, yaw, pose=None):
    # R = np.array([[ np.cos(yaw), np.sin(yaw), 0], 
    #               [-np.sin(yaw), np.cos(yaw), 0], 
    #               [           0,           0, 1]])

    # x_corners = [dx/2, dx/2, -dx/2, -dx/2,
    #              dx/2, dx/2, -dx/2, -dx/2]

    # y_corners = [dy/2, -dy/2, -dy/2, dy/2,
    #              dy/2, -dy/2, -dy/2, dy/2]

    # z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
    #              dz/2, dz/2, dz/2, dz/2]
    

    R = np.array([[ np.cos(yaw), -np.sin(yaw), 0], 
                  [np.sin(yaw), np.cos(yaw), 0], 
                  [           0,           0, 1]])


    x_corners = [dx/2, -dx/2, -dx/2, dx/2,
                 dx/2, -dx/2, -dx/2, dx/2]

    y_corners = [-dy/2, -dy/2, dy/2, dy/2,
                 -dy/2, -dy/2, dy/2, dy/2]
    z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
                 dz/2, dz/2, dz/2, dz/2]
                     
    xyz = np.vstack([x_corners, y_corners, z_corners])
    corners_3d_cam2 = np.zeros((4,8),dtype=np.float32)
    corners_3d_cam2[-1] = 1
    # print(xyz)
    corners_3d_cam2[:3] = np.dot(R, xyz)
    corners_3d_cam2[0,:] += x
    corners_3d_cam2[1,:] += y
    corners_3d_cam2[2,:] += z
    # print(corners_3d_cam2.shape)

    if pose is not None:
        pose = np.matrix(pose)
        corners_3d_cam2 = np.matmul(pose.I, corners_3d_cam2)

    return corners_3d_cam2[:3]


def get_pkl_info(pkl_path):
    data = pickle.load(open(pkl_path, 'rb'))
    return data['scene_name']


def get_all_det_pkl(pkl_path):
    data = pickle.load(open(pkl_path, 'rb'))
    return data


def get_dets_fname(data, frame_name, thres=0.5):
    # frame_name ： 'seq_0_frame_0.pkl'
    box3d_lidar = data[frame_name]['box3d_lidar']    
    scores = data[frame_name]['scores']    
    label_preds = data[frame_name]['label_preds']
    mask = scores>thres

    return np.array(box3d_lidar[mask]), np.array(label_preds[mask]), np.array(scores[mask])

    # mask2 = box3d_lidar[:, 1] < 15  # -5
    # mask3 = box3d_lidar[:, 1] > 0  # -15
    # mask4 = box3d_lidar[:, 0] > 0  # 25
    # mask5 = box3d_lidar[:, 0] < 10  # 26
    # maskAll = mask * mask2 * mask3 * mask4 * mask5
    # return np.array(box3d_lidar[maskAll]), np.array(label_preds[maskAll]), np.array(scores[maskAll])

    
def display_laser_on_image(img, pcl, vehicle_to_image, pcl_attr=None):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl, np.ones_like(pcl[:,0:1])), axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1) 

    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:,2] > 0
    proj_pcl = proj_pcl[mask]
    if pcl_attr is None:
        pcl_attr = np.ones_like(pcl[:,0:1])
    proj_pcl_attr = pcl_attr[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
        np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]

    # Colour code the points based on attributes (distance/intensity...)
    coloured_intensity = 255*cmap(proj_pcl_attr[:,0]/30)
    # print(coloured_intensity)

    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i])


def get_vehicle_to_image(matrix_path = ""):
    # this_tf_prefix =  'segment-%s_with_camera_labels' % (get_pkl_info(point_lidar_path))  # 找到对应当前 frame 序列的 tfrecord

    cam_front_ex_matrix = np.loadtxt(matrix_path + "_cam_front_ex_matrix.txt").reshape(4, 4) 
    cam_front_in_matrix = np.loadtxt(matrix_path + "_cam_front_in_matrix.txt")

    vehicle_to_image = transform.get_image_transform(cam_front_ex_matrix, cam_front_in_matrix)
    return vehicle_to_image


def get_registration_angle(mat):
    # cos_theta = mat[0, 0]
    # sin_theta = mat[1, 0]

    # if cos_theta < -1:
    #     cos_theta = -1
    # if cos_theta > 1:
    #     cos_theta = 1

    # theta_cos = np.arccos(cos_theta)

    # if sin_theta >= 0:
    #     return theta_cos
    # else:
    #     return 2 * np.pi - theta_cos

    return np.arctan2(-mat[1, 0], mat[0, 0])


def veh_to_world_bbox(boxes, pose):
    # ang = get_registration_angle(pose)
    ang = np.arctan2(-pose[1, 0], pose[0, 0])
    ones = np.ones(shape=(boxes.shape[0], 1))
    # for i in range(t_id):
    b_id = 0
    box_xyz = boxes[:, b_id:b_id + 3]
    box_xyz1 = np.concatenate([box_xyz, ones], -1)
    
    box_world = np.matmul(box_xyz1, pose.T)

    # print(pose)
    # print("orig shift: ", np.linalg.norm(np.matmul(np.array([[0,0,0,1]]), pose.T)[:,:3], ord=2))

    # print('mean shift , ',((box_world[:,:3]-box_xyz1[:,:3])**2).mean())
    # box_world = box_xyz1
    boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
    boxes[:, b_id + 6] += ang

    # 角度约束在 0 - 2*pi 
    for box in boxes:
        while box[6] < 0:
            box[6] += 2 * np.pi
        while box[6] > 2 * np.pi:
            box[6] -= 2 * np.pi

    return boxes


def world_to_veh_bbox(boxes, pose):
    # ang = get_registration_angle(pose)
    ang = np.arctan2(-pose[1, 0], pose[0, 0])
    ones = np.ones(shape=(boxes.shape[0], 1))


    center_world = boxes[:, :3]
    center_world1 = np.concatenate([center_world, ones], -1)
    centers_xyz = np.matmul(center_world1, np.matrix(pose.T).I)

    boxes[:, :3] = centers_xyz[:, 0:3]
    boxes[:, 6] -= ang

    # 角度约束在 0 - 2*pi 
    for box in boxes:
        while box[6] < 0:
            box[6] += 2 * np.pi
        while box[6] > 2 * np.pi:
            box[6] -= 2 * np.pi

    return boxes


if __name__ == "__main__":

    ############################################
    #########  图片保存到目录下，画上 box  #########
    ############################################
    SEQ_IDX = 0
    D_PATH = '/mnt/data/WAYMO_det3d/'
    TF_PATH = D_PATH + 'tfrecord_validation/segment-%s_with_camera_labels.tfrecord'
    PKL_LABEL = D_PATH + 'val/annos/seq_%d_frame_0.pkl'%SEQ_IDX
    IMG_SAVE_PATH = D_PATH + "val/images/"
    # IMG_SAVE_PATH = "/home/zhanghao/wimages/"

    TF_this_seq = TF_PATH % (get_pkl_info(PKL_LABEL))  # 找到对应当前 frame 序列的 tfrecord
    
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

    # serialize_image_tfrecord(TF_this_seq, IMG_SAVE_PATH, 0, "seq_%s_frame_"%SEQ_IDX)
    serialize_5image_tfrecord(TF_this_seq, IMG_SAVE_PATH, "seq_%s_frame_"%SEQ_IDX)

    ############################################
    ############################################


