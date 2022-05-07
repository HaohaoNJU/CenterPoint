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
from torch import double, nn
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss, BinRotLoss
from det3d.models.utils import Sequential, get_binrot_alpha, get_binrot_target,_nms,_circle_nms
from ..registry import HEADS
from .utils import SepHead,DCNSepHead
from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from ...core.utils.center_utils import _transpose_and_gather_feat
import copy 


@HEADS.register_module
class CenterHead(nn.Module):
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
        iou_cfg = dict(),
    ):
        super(CenterHead, self).__init__()

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
        self.is_iou_aux = common_heads.get("iou",False)
        if self.is_iou_aux:
            self.iou_weight = iou_cfg.get('weight',0)
            self.iou_power = torch.Tensor(iou_cfg["power"])

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        self.crit_rot = BinRotLoss()
        self.train_cfg = {}

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

    @torch.no_grad()
    def _iou_target(self, example, preds_dict, task_id):
        batch, _, H, W = preds_dict['hm'].size()
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, 1, H, W).repeat(batch, 1, 1, 1).to(preds_dict['hm'])
        xs = xs.view(1, 1, H, W).repeat(batch, 1, 1, 1).to(preds_dict['hm'])

        if self.is_bin_rot:
            batch_det_rot = get_binrot_alpha(preds_dict['rot'],channel_first=True) # (B,1,H,W)
        else:
            batch_det_rots = preds_dict['rot'][:, 0:1, :, :]
            batch_det_rotc = preds_dict['rot'][:, 1:2, :, :]
            batch_det_rot = torch.atan2(batch_det_rots, batch_det_rotc)

        batch_det_dim = torch.exp(preds_dict['dim'])
        batch_det_reg = preds_dict['reg']
        batch_det_hei = preds_dict['height']
        batch_det_xs = xs + batch_det_reg[:, 0:1, :, :]
        batch_det_ys = ys + batch_det_reg[:, 1:2, :, :]
        batch_det_xs = batch_det_xs * self.train_cfg.out_size_factor * self.train_cfg.voxel_size[0] + self.train_cfg.pc_range[0]
        batch_det_ys = batch_det_ys * self.train_cfg.out_size_factor * self.train_cfg.voxel_size[1] + self.train_cfg.pc_range[1]

        # (B, 7, H, W)
        batch_box_preds = torch.cat([batch_det_xs, batch_det_ys, batch_det_hei, batch_det_dim, batch_det_rot], dim=1)
        batch_box_preds = _transpose_and_gather_feat(batch_box_preds, example['ind'][task_id])
        target_box = example['anno_box'][task_id]
        batch_gt_dim = torch.exp(target_box[..., 3:6])
        batch_gt_reg = target_box[..., 0:2]
        batch_gt_hei = target_box[..., 2:3]
        batch_gt_rot = torch.atan2(target_box[..., -2:-1], target_box[..., -1:])
        batch_gt_xs = _transpose_and_gather_feat(xs, example['ind'][task_id]) + batch_gt_reg[..., 0:1]
        batch_gt_ys = _transpose_and_gather_feat(ys, example['ind'][task_id]) + batch_gt_reg[..., 1:2]
        batch_gt_xs = batch_gt_xs * self.train_cfg.out_size_factor * self.train_cfg.voxel_size[0] + self.train_cfg.pc_range[0]
        batch_gt_ys = batch_gt_ys * self.train_cfg.out_size_factor * self.train_cfg.voxel_size[1] + self.train_cfg.pc_range[1]
        # (B, max_obj, 7)
        batch_box_targets = torch.cat([batch_gt_xs, batch_gt_ys, batch_gt_hei, batch_gt_dim, batch_gt_rot], dim=-1)
        iou_targets = boxes_iou3d_gpu(batch_box_preds.reshape(-1, 7), batch_box_targets.reshape(-1, 7))[range(
            batch_box_preds.reshape(-1, 7).shape[0]), range(batch_box_targets.reshape(-1, 7).shape[0])]
        return iou_targets.reshape(batch, -1, 1)



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
        
        # box_loss *= self.code_weights[0]
        reg_loss = box_loss[:2].mean() * self.code_weights[0]
        hei_loss = box_loss[2] * self.code_weights[1]
        dim_loss = box_loss[3:6].mean() * self.code_weights[2]
        rot_loss = rot_loss * self.code_weights[3]
        # loc_loss = (box_loss*box_loss.new_tensor(self.code_weights[:-8])).sum()
        # loc_loss = box_loss.sum() + rot_loss
        loc_loss = reg_loss + hei_loss + dim_loss + rot_loss
        return loc_loss, (reg_loss , hei_loss , dim_loss , rot_loss)

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
                # box_loss, rot_loss, loc_loss = self._compute_binrot_loc_loss(preds_dict,target_box,example,task_id)
                loc_loss, (reg_loss , hei_loss , dim_loss , rot_loss) = self._compute_binrot_loc_loss(preds_dict,target_box,example,task_id)
            else : 
                box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)
                reg_loss = box_loss[:2].mean() * self.code_weights[0]
                hei_loss = box_loss[2] * self.code_weights[1]
                dim_loss = box_loss[3:6].mean() * self.code_weights[2]
                rot_loss = box_loss[6:8].mean() * self.code_weights[3]
                loc_loss = reg_loss + hei_loss + dim_loss + rot_loss
            loss = hm_loss + self.weight*loc_loss
            if self.is_iou_aux and self.iou_weight > 0 :
                iou_targets = self._iou_target(example, preds_dict, task_id)
                preds_dict['iou'] = torch.clamp(preds_dict['iou'],min=0, max=1)
                iou_loss = self.crit_reg(preds_dict['iou'], example['mask'][task_id], example['ind'][task_id], iou_targets)
                loss += self.iou_weight * iou_loss.sum() 
                ret.update({'iou_loss' : iou_loss.detach().cpu() })
                # TODO TODO TODO TODO : Cut here ! ! !
            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, \
            'reg_loss' : reg_loss, 'hei_loss' : hei_loss, 'dim_loss' : dim_loss, 'rot_loss' : rot_loss, \
            'num_positive': example['mask'][task_id].float().sum()})
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

            if self.is_bin_rot:
                batch_rot = get_binrot_alpha(preds_dict['rot'])
            else:
                batch_rots = preds_dict['rot'][..., 0:1]
                batch_rotc = preds_dict['rot'][..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            # batch_dim = preds_dict['dim']
            batch_dim = torch.exp(preds_dict['dim'])
            batch, H, W, _ = batch_reg.size()
            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W, 1).repeat(batch, 1, 1, 1).to(batch_reg)
            xs = xs.view(1, H, W, 1).repeat(batch, 1, 1, 1).to(batch_reg)
            xs = xs + batch_reg[:, :, :, 0:1]
            ys = ys + batch_reg[:, :, :, 1:2]
            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            batch, H, W, num_cls = batch_hm.size()
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            if self.is_iou_aux and self.iou_weight > 0:
                batch_iou = preds_dict['iou']
                batch_iou = batch_iou.reshape(batch, H*W, 1).repeat(1,1,num_cls)
                self.iou_power = self.iou_power.to(batch_iou)
                batch_hm = torch.pow(batch_hm, 1-self.iou_power) * torch.pow(batch_iou, self.iou_power)  

            xs = xs.reshape(batch, H*W, 1)
            ys = ys.reshape(batch, H*W, 1)
            batch_hei = batch_hei.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_rot = batch_rot.reshape(batch,H*W, 1)
            #############################################################################################


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



