
import numpy as np
import torch 
from det3d.core.utils.circle_nms_jit import circle_nms

def get_pred_depth(depth):
  return depth

def get_binrot_alpha(rot, channel_first=False):
    
  # output: (...,B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[..., 0]
  assert (len(rot) == 4, "Tensor rot need to have 4 dims")
  if isinstance(rot, torch.Tensor):
    if channel_first:
      alpha1 = torch.atan2(rot[:,2:3,:,:], rot[:,3:4,:,:]) + (-0.5 * np.pi)
      alpha2 = torch.atan2(rot[:,6:7,:,:], rot[:,7:8,:,:]) + ( 0.5 * np.pi)
    else:
      alpha1 = torch.atan2(rot[..., 2:3], rot[..., 3:4]) + (-0.5 * np.pi)
      alpha2 = torch.atan2(rot[..., 6:7], rot[..., 7:8]) + ( 0.5 * np.pi)
  # elif isinstance(rot, np.ndarray):
  #   alpha1 = np.arctan2(rot[..., 2], rot[..., 3]) + (-0.5 * np.pi)
  #   alpha2 = np.arctan2(rot[..., 6], rot[..., 7]) + ( 0.5 * np.pi)
  else:
      raise TypeError("Tensor rot dtype is invalid ! ")
  if channel_first:
    idx = rot[:,1:2,:,:] > rot[:,5:6,:,:]
  else:
    idx = rot[..., 1:2] > rot[..., 5:6]
  idx = idx.int()
  alpha = alpha1 * idx + alpha2 * (1-idx)
  alpha[alpha<-np.pi] += 2* np.pi
  alpha[alpha>np.pi] -= 2*np.pi
  return alpha

def get_binrot_target(target_sin, target_cos):
    theta = torch.atan2(target_sin, target_cos)
    # clip theta value into [-pi, pi]
    theta_bin1 = theta + np.pi /2 
    theta_bin2 = theta - np.pi /2
    theta_bin1[theta_bin1>np.pi] -= np.pi * 2
    theta_bin2[theta_bin2<-np.pi] += np.pi * 2

    cls_bin1 = torch.ge(theta, 5*np.pi / 6) * torch.le(theta, np.pi) + torch.ge(theta, -np.pi) * torch.le(theta, np.pi/6)
    cls_bin2 = torch.ge(theta, -np.pi / 6) * torch.le(theta, np.pi) + torch.ge(theta, -np.pi) * torch.le(theta, -5 * np.pi/6)
    cls_bin1 = cls_bin1.long()
    cls_bin2 = cls_bin2.long()
    target_res = torch.stack([theta_bin1, theta_bin2],dim =-1 )
    target_bin = torch.stack([cls_bin1, cls_bin2],dim =-1 )
    return target_bin, target_res




def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
    keep = torch.from_numpy(keep).long().to(boxes.device)
    return keep  
# TODO Add fast nms
def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2
  hmax = nn.functional.max_pool2d(
      heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep