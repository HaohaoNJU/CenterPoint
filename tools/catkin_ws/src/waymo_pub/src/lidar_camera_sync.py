from email.utils import collapse_rfc2231_value
from re import L
import cv2
import pickle

from matplotlib.pyplot import axis
import open3d
import numpy as np
from numba import jit
import matplotlib
import matplotlib.cm
matplotlib.use("cairo")
cmap = matplotlib.cm.get_cmap("viridis")


class LidarCameraSync(object) :
    def __init__(self,extrinsic,intrinsic):
        self.lidar_to_image = None
        self.set_proj_param(extrinsic, intrinsic)

        self.LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
        self.LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
        self.LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
        # self.LINES+= [[1, 6], [2, 5]] # front face and draw x
        self.LINES += [ [0,7],[3,4]  ]

    def set_proj_param(self, extrinsic, intrinsic):
        assert extrinsic.shape==(16,) or extrinsic.shape ==(4,4), "Extrinsic shape error, you should pass an array with total length of 16"
        assert intrinsic.shape==(9,) or intrinsic.shape ==(3,3), "Intrinsic shape error, you should pass an array with total length of 9"
        extrinsic = extrinsic.reshape(4,4)
        intrinsic = np.concatenate( (intrinsic.reshape(3,3),np.zeros((3,1),dtype=np.float32) ), axis = -1 )
        self.lidar_to_image = intrinsic.dot(extrinsic)


    def proj_point_to_image(self, points, img, show = True):
        xyz1 = np.concatenate((points[:,:3], np.ones_like(points[:,0:1])), axis=1)
        # print(pcl1)
        # Transform the point cloud to image space.
        proj_pcl = np.einsum('ij,bj->bi', self.lidar_to_image, xyz1) 
        # print(proj_pcl)
        # Filter LIDAR points which are behind the camera.
        mask = proj_pcl[:,2] > 0
        proj_pcl = proj_pcl[mask]
        xyz1 = xyz1[mask]
        # Project the point cloud onto the image.
        proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]
        # Filter points which are outside the image.
        mask = np.logical_and(
            np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
            np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[0]))
        proj_pcl = proj_pcl[mask]

        # Draw a circle for each point.
        if show : 
            xyz1 = xyz1[mask]
            proj_pcl_attr = np.sqrt(np.sum(xyz1[:,:3]**2, axis = -1) )
            # Colour code the points based on attributes (distance/intensity...)
            coloured_intensity = 255*cmap(proj_pcl_attr / 80)
#         coloured_intensity = rgba(proj_pcl_attr)
            for i in range(proj_pcl.shape[0]):
                cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 3, coloured_intensity[i], -1)
        return img, proj_pcl


    def proj_box_to_image(self, bboxs, img, color = [255,0,255], rand_color=False, all_in = True, show  = True, format_2d = False, texts = None):
        assert texts is None or  len(texts) == len(bboxs), "Texts should have the same elements as bboxs ! "
        if not rand_color:
            multi_color = False
            if isinstance(color, np.ndarray) and color.shape == (len(bboxs),3):
                multi_color = True
            elif isinstance(color, np.ndarray):
                color = color.tolist()
        
        corners8_projs = [];bbox_ids = []
        for boxid, bbox in enumerate(bboxs):
            corners8 = self._compute_3dcornors(bbox) # (N, 3)
            if rand_color:
                c = np.random.randint(0,255,3).tolist()
            elif multi_color:
                c = color[boxid].tolist()
            else:
                c = color
            text = None if texts is None else texts[boxid]
            _, corners8_proj = self._display_3dbox_on_image(img, corners8,c, all_in = all_in, show = show, format_2d = format_2d, text = text)

            if  not show and corners8_proj is not None:
                corners8_projs.append(corners8_proj)
                bbox_ids.append(boxid)        
        if show :
            return img, None
        elif len(bbox_ids) > 0 : 
            return np.array(bbox_ids), np.stack(corners8_projs, axis=0)
        else:
            return None, None

    def _display_3dbox_on_image(self, img, corners8, color = [255, 0, 255], all_in = True, show = True, format_2d = None,  text = None):
        # print(corners8)

        corners8_1 = np.concatenate((corners8, np.ones((8,1))), axis=1)
        corners8_proj = np.einsum('ij,bj->bi', self.lidar_to_image, corners8_1)
        # print(corners8_proj)

        mask = corners8_proj[:,2] > 0
        if not mask.all() > 0:
            return None, None
        # Project the points onto the image.
        corners8_proj = corners8_proj[:,:2] / corners8_proj[:,2:3]
        # Filter points which are outside the image.

        mask = np.logical_and(
            np.logical_and(corners8_proj[:,0] > 0, corners8_proj[:,0] < img.shape[1]),
            np.logical_and(corners8_proj[:,1] > 0, corners8_proj[:,1] < img.shape[0]))

        if  all_in and  mask.any() <= 0:
            return  None, None
        elif  (not all_in) and mask.sum() <= 0:
            # print(mask)
            return None, None
    
        if format_2d:
            box2d = self._compute_2dbox(corners8_proj, img.shape)
            top_left = int(box2d[0]), int(box2d[1])
            bottom_right = int(box2d[2]), int(box2d[3])
            cv2.rectangle(img, top_left, bottom_right, color, 6)
            if text is not None:
                place = (top_left[0], top_left[1]-5)
                cv2.putText(img, text =  text, org = place,  fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.0, color = color, thickness = 1)
        elif show:
            # Draw a circle for each point.
            for i in range(corners8_proj.shape[0]):
                cv2.circle(img, (int(corners8_proj[i,0]),int(corners8_proj[i,1])), 5, color, -1)
                # print("--"*20)

            for l in self.LINES:
                p1 = (int(corners8_proj[l[0], 0]), int(corners8_proj[l[0], 1]))
                p2 = (int(corners8_proj[l[1], 0]), int(corners8_proj[l[1], 1]))
                cv2.line(img, p1, p2, color , 2, cv2.LINE_8)
            
            if text is not None:
                place = (corners8_proj[:,0].min(), corners8_proj[:,1].min()-5)
                cv2.putText(img, text =  text, org = place,  fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.0, color = color, thickness = 1)
        return img, corners8_proj

    def _compute_3dcornors(self, bbox, pose=None):
        x, y, z, dx, dy, dz, yaw = bbox[:7]
        R = np.array([[ np.cos(yaw), -np.sin(yaw), 0], 
                      [np.sin(yaw), np.cos(yaw), 0], 
                      [           0,           0, 1]])

        x_corners = [dx/2, -dx/2, -dx/2, dx/2,
                     dx/2, -dx/2, -dx/2, dx/2]
    
        y_corners = [-dy/2, -dy/2, dy/2, dy/2,
                     -dy/2, -dy/2, dy/2, dy/2]
        z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
                     dz/2, dz/2, dz/2, dz/2]

        # R = np.array([[ np.cos(yaw), np.sin(yaw), 0], 
        #               [-np.sin(yaw), np.cos(yaw), 0], 
        #               [           0,           0, 1]])
    
        # x_corners = [dx/2, dx/2, -dx/2, -dx/2,
        #              dx/2, dx/2, -dx/2, -dx/2]
    
        # y_corners = [dy/2, -dy/2, -dy/2, dy/2,
        #              dy/2, -dy/2, -dy/2, dy/2]
    
        # z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
        #          dz/2, dz/2, dz/2, dz/2]

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
        return corners_3d_cam2[:3].T # (N , 3)

    def _compute_2dbox(self, points, img_shape):
        """
        @points : shape (..., 8, 2+)
        """
        # Compute the 2D bounding box and draw a rectangle
        x1 = np.amin(points[...,0], axis = -1)
        x2 = np.amax(points[...,0], axis = -1)
        y1 = np.amin(points[...,1], axis = -1)
        y2 = np.amax(points[...,1], axis = -1)
        # x1 = min(max(0,x1),img_shape[1])
        # x2 = min(max(0,x2),img_shape[1])
        # y1 = min(max(0,y1),img_shape[0])
        # y2 = min(max(0,y2),img_shape[0])
        x1  = np.clip(x1, a_min = 0, a_max = img_shape[1])
        x2  = np.clip(x2, a_min = 0, a_max = img_shape[1])
        y1  = np.clip(y1, a_min = 0, a_max = img_shape[0])
        y2  = np.clip(y2, a_min = 0, a_max = img_shape[0])
        return np.stack([x1,y1,x2,y2], axis = -1)





################################################################################################################################################################
#############################                                                                                                                OLD                           FUNCTIONS                                                                                                           ##################
################################################################################################################################################################





# connect vertic
LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
# LINES+= [[1, 6], [2, 5]] # front face and draw x





def draw_bbox(img, bboxes):
    for box in bboxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    return img


def read_detections(det_pkl):
    all_dets = pickle.load(open(det_pkl, "rb"))
    return all_dets


def get_dets_fname(all_data_dict, fname, thres=0.5, keys=['boxes', 'scores', 'classes']):
    # fname ： 'xxx.pcd'/'xxx.pkl'
    box3d_lidar = all_data_dict[fname][keys[0]]    
    scores      = all_data_dict[fname][keys[1]]    
    label_preds = all_data_dict[fname][keys[2]]
    mask = scores > thres

    return np.array(box3d_lidar[mask]), np.array(label_preds[mask]), np.array(scores[mask])

    # mask2 = box3d_lidar[:, 1] < 15  # -5
    # mask3 = box3d_lidar[:, 1] > 0  # -15
    # mask4 = box3d_lidar[:, 0] > 0  # 25
    # mask5 = box3d_lidar[:, 0] < 10  # 26
    # maskAll = mask * mask2 * mask3 * mask4 * mask5
    # return np.array(box3d_lidar[maskAll]), np.array(label_preds[maskAll]), np.array(scores[maskAll])


def read_pcd(file_path, padding=0):
    pcd = open3d.io.read_point_cloud(file_path)
    pcd_npy = np.asarray(pcd.points)
    if padding > 0:
        pcd_npy = np.append(pcd_npy, np.zeros((pcd_npy.shape[0], padding)), axis=1)
    return pcd_npy

def get_cv_image(img_path):
    img = cv2.imread(img_path)
    return np.empty(0) if img is None else img



def display_3dbox_on_image(img, corners8, lidar_to_image):
    # print(corners8)
    corners8_1 = np.concatenate((corners8, np.ones((8,1))), axis=1)
    corners8_proj = np.einsum('ij,bj->bi', lidar_to_image, corners8_1)
    # print(corners8_proj)
    mask = corners8_proj[:,2] > 0
    if not mask.all() > 0:
        return
    
    # Project the points onto the image.
    corners8_proj = corners8_proj[:,:2] / corners8_proj[:,2:3]
    
    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(corners8_proj[:,0] > 0, corners8_proj[:,0] < img.shape[1]),
        np.logical_and(corners8_proj[:,1] > 0, corners8_proj[:,1] < img.shape[1]))
    if not mask.all() > 0:
        return

    # Draw a circle for each point.
    for i in range(corners8_proj.shape[0]):
        cv2.circle(img, (int(corners8_proj[i,0]),int(corners8_proj[i,1])), 5, (0,0,255), -1)
        # print("--"*20)
    
    for l in LINES:
        p1 = (int(corners8_proj[l[0], 0]), int(corners8_proj[l[0], 1]))
        p2 = (int(corners8_proj[l[1], 0]), int(corners8_proj[l[1], 1]))
        cv2.line(img, p1, p2, (255,0,255), 2, cv2.LINE_8)




def display_laser_on_image(img, pcl, lidar_to_image, pcl_attr=None):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl, np.ones_like(pcl[:,0:1])), axis=1)
    # print(pcl1)
    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', lidar_to_image, pcl1) 
    # print(proj_pcl)
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
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i], -1)


    def _compute_3dcornors(self, bbox, pose=None):
        x, y, z, dx, dy, dz, yaw = bbox[:7]
        R = np.array([[ np.cos(yaw), -np.sin(yaw), 0], 
                      [np.sin(yaw), np.cos(yaw), 0], 
                      [           0,           0, 1]])
        x_corners = [dx/2, -dx/2, -dx/2, dx/2,
                     dx/2, -dx/2, -dx/2, dx/2]
    
        y_corners = [-dy/2, -dy/2, dy/2, dy/2,
                     -dy/2, -dy/2, dy/2, dy/2]
        z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
                     dz/2, dz/2, dz/2, dz/2]

        # R = np.array([[ np.cos(yaw), np.sin(yaw), 0], 
        #               [-np.sin(yaw), np.cos(yaw), 0], 
        #               [           0,           0, 1]])
    
        # x_corners = [dx/2, dx/2, -dx/2, -dx/2,
        #              dx/2, dx/2, -dx/2, -dx/2]
    
        # y_corners = [dy/2, -dy/2, -dy/2, dy/2,
        #              dy/2, -dy/2, -dy/2, dy/2]
    
        # z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
        #          dz/2, dz/2, dz/2, dz/2]

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
        return corners_3d_cam2[:3].T # (N , 3)



#########################################################################################
### pc 3D-bbox -> image 2D-bbox
#########################################################################################


def get_box_transformation_matrix(box):
    """
    Create a transformation matrix for a given box pose.
    """

    # tx,ty,tz = box.center_x,box.center_y,box.center_z
    tx,ty,tz = box[0], box[1], box[2]
    c = np.cos(box[6])
    s = np.sin(box[6])

    sl, sw, sh = box[3], box[4], box[5]    # 这里如果读取的是 det3d 的 det box，注意顺序

    return np.array([
        [ sl*c, -sw*s,  0, tx],
        [ sl*s,  sw*c,  0, ty],
        [    0,     0, sh, tz],
        [    0,     0,  0,  1]])


def get_3d_box_projected_corners(vehicle_to_image, boxes7d):
    """
    Get the 2D coordinates of the 8 corners of a label's 3D bounding box.
    args:
        vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
        boxes7d: The object label, numpy (7)
    """

    box = boxes7d

    # Get the vehicle pose
    box_to_vehicle = get_box_transformation_matrix(box)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)

    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)
    return vertices


def compute_2d_bounding_box(img_or_shape, points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    if isinstance(img_or_shape, tuple):
        shape = img_or_shape
    else:
        shape = img_or_shape.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)





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

    





