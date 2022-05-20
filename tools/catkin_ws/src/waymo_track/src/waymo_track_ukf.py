#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort

from data_utils import *
from publish_utils import *
import glob
# from sort_3d import Sort
# from sort_3d_wh import Sort
from sort_3d_ukf import  Sort

ROS_RATE = 6
THRESH = 0.5
SEQ_IDX = 107    # 20:很多人/有卡车
TRACK = True

LABEL2CLASSNAME =  {0 : "Car", 1 : "Pedestrian",2:"Cyclist"}
SAVE_PATTERN = "/home/wanghao/Desktop/projects/WaymoVisualize/catkin_ws/src/waymo_pub/src/zhanghao_mota/demo/tk_ukf/seq_%s_frame_%d.txt"
# rp = "/home/wanghao/Desktop/projects/CenterPoint/tensorrt/data/centerpoint_pp_baseline_score0.1_nms0.7_gpuint8/"
# rp = "/home/wanghao/Desktop/projects/CenterPoint/tensorrt/data/centerpoint_pp_baseline_score0.1_nms0.7_gpu/"
# rp = "/home/wanghao/Desktop/projects/CenterPoint/tools/pp_onestage_baseline_torch/pkls/"
# rp = "/home/wanghao/Desktop/projects/CP_TRT/data/centerpoint_pp_baseline_score0.1_nms0.7_gpufp16/"

# rp = "/home/wanghao/Desktop/projects/CP_TRT/data/centerpoint_pp_baseline_score0.1_nms0.7_gpufp16/"
rp =  "/home/wanghao/Desktop/projects/CenterPoint/results/voxelnet_2sweep_3x_withvelo_baseline/pkls/"

# rp = "/home/wanghao/Desktop/projects/CP_TRT/data/centerpoint_pp_baseline_score0.1_nms0.7_gpu/"
# rp = "/home/wanghao/Desktop/projects/CP_TRT/data/centerpoint_pp_baseline_score0.1_nms0.7_gpuint8/"


def get_dets_fname_cpp(frame_name, prefix=rp, thr=THRESH):
    # print(prefix + frame_name + ".bin.txt")
    box9d = np.loadtxt(prefix + frame_name + ".bin.txt")
    box7ds = box9d[:, [0,1,2,3,4,5,8]]
    classes = box9d[:, -1]
    scores = box9d[:, -2]
    mask = scores > thr
    # print(mask)
    return box7ds[mask], classes[mask], scores[mask]


def get_dets_fname_gt(frame_name, thres=THRESH):
    data = pickle.load(open(frame_name, "rb"))
    # box3d_lidar = data['box3d_lidar']    
    box3d_lidar = data['box3d_lidar'][:, [0,1,2,3,4,5,8] ]
    velocities = data['box3d_lidar'][:,[6,7]]
    scores = data['scores']    
    label_preds = data['label_preds']
    mask = scores>thres

    return np.array(box3d_lidar[mask]), np.array(label_preds[mask]), np.array(scores[mask]), np.array(velocities[mask])



def get_veh_to_global(frame_name) :
    data = pickle.load(open(frame_name, "rb"))
    RT = data['veh_to_global'].reshape(4,4)
    return RT

def transform_bbox(box3d_preds, rt):
    box3d_preds[:,:3] = box3d_preds[:,:3].dot(rt[:3,:3].T) + rt[:3,-1]
    cos_theta = np.cos(box3d_preds[:,6])
    sin_theta = np.sin(box3d_preds[:,6])
    n_ = len(sin_theta)
    rot_z = np.stack( [cos_theta, -sin_theta, np.zeros(n_),   sin_theta, cos_theta, np.zeros(n_),   np.zeros(n_),np.zeros(n_),np.ones(n_) ],axis=0).reshape(3,3,-1)
    mat_mul = rt[:3,:3].dot(rot_z)
    box3d_preds[:,6] = np.arctan2(mat_mul[1,0], mat_mul[1,1])
    return box3d_preds

def waymo_veh_to_world_bbox(boxes, pose):
    # ang = get_registration_angle(pose)
    # get rot angle  from clockwise
    ang = np.arctan2(pose[1, 0], pose[0, 0])
    ones = np.ones(shape=(boxes.shape[0], 1))
    # for i in range(t_id):
    b_id = 0
    box_xyz = boxes[:, b_id:b_id + 3]
    box_xyz1 = np.concatenate([box_xyz, ones], -1)
    box_world = np.matmul(box_xyz1, pose.T)
    boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
    boxes[:, b_id + 6] += ang

    # 角度约束在 0 - 2*pi 
    for box in boxes:
        while box[6] < 0:
            box[6] += 2 * np.pi
        while box[6] > 2 * np.pi:
            box[6] -= 2 * np.pi
    return boxes

def kitti2waymo(bbox):
    bbox[:,6] = -(bbox[:,6] + np.pi /2 )
    return bbox[:, [0,1,2,4,3,5, 6]]
    

if  __name__ == "__main__":


    # LIDAR_PATH = "/home/zhanghao/seq201_to_zh/lidar/seq_%s_frame_"%SEQ_IDX
    LIDAR_PATH = "/mnt/data/waymo_opensets/val/lidar/seq_%s_frame_"%SEQ_IDX
    ANNO_PATH = "/mnt/data/waymo_opensets/val/annos/seq_%s_frame_"%SEQ_IDX

    frame_nums = len(glob.glob(LIDAR_PATH+"*"))
    print(frame_nums)

    frame = 0
    bridge = CvBridge()
    rospy.init_node('waymo_node',anonymous=True)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)
    
    rate = rospy.Rate(ROS_RATE)
    ################################################################################
    if TRACK:
        mot_tracker = Sort(max_age= 10, 
                           min_hits=3,
                           iou_threshold=0.5)
        
    # [x,y,z,dx,dy,dz,r,score,class
    ################################################################################
    while not rospy.is_shutdown():
        # image = get_image_tfrecord(TF_PATH, frame)
        point_cloud = get_lidar_pkl(LIDAR_PATH + "%s.pkl"%frame)

        # box3d_preds, label_preds, scores = get_dets_fname_cpp("seq_%s_frame_%s"%(SEQ_IDX, frame))
        box3d_preds, label_preds, scores, velocities = get_dets_fname_gt(rp + "seq_%s_frame_%s.pkl"%(SEQ_IDX, frame))
        box3d_preds = kitti2waymo(box3d_preds)
        if TRACK : 
            # [x,y,z,dx,dy,dz,r,score,class
            
            velocities = np.concatenate( [velocities, np.zeros((len(velocities),1))  ], axis = -1 )
            vel_to_global = get_veh_to_global(ANNO_PATH + "%s.pkl"%frame)
            box3d_preds = waymo_veh_to_world_bbox(box3d_preds, vel_to_global)
            velocities = velocities.dot(vel_to_global[:3,:3].T)
            box3d_preds = np.concatenate([box3d_preds,scores.reshape(-1,1), label_preds.reshape(-1,1)],axis = -1)
            box3d_preds = mot_tracker.update(box3d_preds, velocities= velocities)
        
        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)
        
        if TRACK:
            vel_to_global_inv = np.linalg.inv(vel_to_global) 
            rot_inv = np.linalg.inv(vel_to_global[:3,:3])
            # box3d_preds = transform_bbox(box3d_preds, np.linalg.inv(vel_to_global) )
            #  12 dof (x,y,z,dx,dy,dz,yaw,score, label, vx, vy,  id)
            box3d_preds[:,:7] = waymo_veh_to_world_bbox(box3d_preds[:,:7], vel_to_global_inv )
            # velocities =np.concatenate([ box3d_preds[:,-3:-1] , np.zeros((len(box3d_preds), 1))], axis = -1)
            # box3d_preds[:,-3:-1] = velocities.dot(rot_inv.T)[:,:2]
            ### To save tracking results 
            np.savetxt(SAVE_PATTERN % (SEQ_IDX, frame), 
                                   box3d_preds[:, [-1,-4, 0,1,2,3,4,5,6,7,9,10]] )

            # box3d_preds[:,:7] = kitti2waymo(box3d_preds[:,:7])
            

  
            
        corner_3d_velos = []
        for boxes in box3d_preds:
            # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velos.append(np.array(corner_3d_velo).T)
        
        # if TRACK:
        #     vel_to_global = get_veh_to_global(ANNO_PATH + "%s.pkl"%frame)
        #     corner_3d_velos = np.stack(corner_3d_velos,axis=0).reshape(-1,3)
        #     corner_3d_velos = np.stack([corner_3d_velos,np.ones([len(corner_3d_velos),1])] , axis=-1 ).T # 4 x n
        #     corner_3d_velos = vel_to_global.dot(corner_3d_velos)
            
            

        # classnames = [LABEL2CLASSNAME.get(x, "Unknown") for x in label_preds]
        scores = np.around (scores, 3)
        
        if TRACK :
            scores = np.around (box3d_preds[:,7], 3)
            texts = ["ID"+str(x) for x in box3d_preds[:,-1].astype(np.int32)]
            print(frame, len(box3d_preds))
            publish_3dbox(box3d_pub, corner_3d_velos, texts=texts, types=box3d_preds[:,-1], track_color=True, Lifetime=1. / ROS_RATE)
        
        else:
            publish_3dbox(box3d_pub, corner_3d_velos, texts=scores, types=label_preds, track_color=False, Lifetime=1. / ROS_RATE)

        rospy.loginfo("waymo published")
        rate.sleep()
        
        frame += 1
        if frame == (frame_nums-1):
            frame = 0


