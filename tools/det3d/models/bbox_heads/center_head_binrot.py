# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------

import logging
from collections import defaultdict
from det3d.core import box_torch_ops
import torch
from det3d.torchie.cnn import kaiming_init
from torch import double, nn
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss, BinRotLoss
from det3d.models.utils import Sequential, get_binrot_alpha, get_binrot_target
from ..registry import HEADS
from . import SepHead,DCNSepHead
import copy 
try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

from det3d.core.utils.circle_nms_jit import circle_nms
import numpy as np 

def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
    keep = torch.from_numpy(keep).long().to(boxes.device)
    return keep  

class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x




@HEADS.register_module
class CenterHeadBinRot(nn.Module):
    def __init__(
        self,
        in_channels=[128,],
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
        is_bin_rot = False,
        is_iou_aux = False,
        
    ):
        super(CenterHeadBinRot, self).__init__()

        # tasks = [ dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']), ]
        
        # TODO num_classes : [3]
        num_classes = [len(t["class_names"]) for t in tasks]
        # TODO class_names  :  ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights 
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.is_bin_rot = is_bin_rot
        self.is_iou_aux = is_iou_aux
        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        self.crit_rot = BinRotLoss()

        if is_bin_rot :
            assert (common_heads['rot'][0] == 8, "Binrot head need to set 8 channels !") 
        # TODO common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(8, 2)}
        # TODO box_n_dim = 7
        self.box_n_dim = 9 if 'vel' in common_heads else 7  
        self.use_direction_classifier = False 

        if not logger:
            logger = logging.getLogger("CenterHeadBinRot")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                #  num_cls = 3, num_hm_conv = 2
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                # share_conv_channel = 64 , init_bias = -2.19
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3))
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3))

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y
    def _compute_binrot_loc_loss(self,preds_dict,target_box,example,task_id):
        # Regression loss (SmoothL1 Loss ) for dimension, offset, height, rotation            
        box_loss = self.crit_reg(preds_dict['anno_box'][:,:6],\
         example['mask'][task_id], example['ind'][task_id], \
         target_box[...,:-2])
    
        # Specially for bin rot loss
        target_bin, target_res = get_binrot_target(target_box[...,-2],target_box[...,-1])
        rot_loss = self.crit_rot(preds_dict['anno_box'][:,6:14],\
         example['mask'][task_id], example['ind'][task_id], 
         target_bin,target_res)
    
        box_loss *= box_loss.new_tensor(self.code_weights[:6])
        rot_loss = rot_loss * self.code_weights[6]
        # loc_loss = (box_loss*box_loss.new_tensor(self.code_weights[:-8])).sum()
        loc_loss = box_loss.sum() + rot_loss
        return box_loss, rot_loss, loc_loss

    # TODO ： For Training  
    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            # heads = {'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 8), 'hm' : (3,2)}
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            # TODO : FastFocalLoss ,defined in CornerNet, see in file models/losses/centernet_loss.py FastFocalLoss
            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['vel'], preds_dict['rot']), dim=1)  
            else:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['rot']), dim=1)  
                target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2,-1]] # remove vel target                       

            ret = {}
 
            if self.is_bin_rot :
                box_loss, rot_loss, loc_loss = self._compute_binrot_loc_loss(preds_dict,target_box,example,task_id)
            else : 
                box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)
                loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()
                rot_loss = box_loss[6:8]
            loss = hm_loss + self.weight*loc_loss
            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu()[:6],\
            'rot_loss' : rot_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})
            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        return rets_merged

    # TODO ： For Inference 
    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []
        double_flip = test_cfg.get('double_flip', False)
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )
        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()
            batch_size = preds_dict['hm'].shape[0]

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]


            batch_hm = torch.sigmoid(preds_dict['hm'])
            batch_dim = torch.exp(preds_dict['dim'])


            if self.is_bin_rot:
                batch_rot = get_binrot_alpha(preds_dict['rot'])
            else:
                batch_rots = preds_dict['rot'][..., 0:1]
                batch_rotc = preds_dict['rot'][..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                pass  # TODO TODO TODO : NEED TO ADD HERE CLS_SPECIFIC NMS
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, task_id)) 

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                else:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
            ret['metadata'] = metas[0][i]
            ret_list.append(ret)
        return ret_list 

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, task_id):
        batch_size = len(batch_hm)
        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            #  batch_hm : (batch, H*W, 3 )
            hm_preds = batch_hm[i]
            scores, labels = torch.max(hm_preds, dim=-1)
            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)
            mask = distance_mask & score_mask 
            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.get('circular_nms', False):
                centers = boxes_for_nms[:, [0, 1]] 
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(boxes, min_radius=test_cfg.min_radius[task_id], post_max_size=test_cfg.nms.nms_post_max_size)  
            else:
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(), 
                                    thresh=test_cfg.nms.nms_iou_threshold,
                                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                    post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]


            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels,
            }
            prediction_dicts.append(prediction_dict)
        return prediction_dicts 










    