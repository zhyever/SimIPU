import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import auto_fp16
from os import path as osp

from mmdet3d.core import Box3DMode, Coord3DMode, show_result
from mmdet.models.detectors import BaseDetector

from mmdet3d.models import builder
from mmdet.core import multi_apply

from mmdet.models import DETECTORS

from torch.nn import functional as F
from mmdet3d.models.fusion_layers.coord_transform import apply_3d_transformation

from torch import nn

from mmdet3d.ops import build_sa_module
from mmdet3d.models.model_utils import VoteModule

from scipy.optimize import linear_sum_assignment

import copy

import numpy as np

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)

class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def forward(self, x, label, topk=(1, 5)):
        bsz = x.shape[0]
        x = x.squeeze()
        loss = self.criterion(x, label)
        acc = self.accuracy(x, label, topk=topk)
        return loss, acc


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

@DETECTORS.register_module()
class Inter_Intro_moco_better(BaseDetector):
    """Base class for detectors."""

    def __init__(self,
                 img_backbone=None,
                 pts_backbone=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_head=None
                 ):
        super(Inter_Intro_moco_better, self).__init__()

        self.train_cfg = train_cfg
        self.cl_cfg = train_cfg["cl_strategy"]

        self.pts_intro_hidden_dim = self.cl_cfg["pts_intro_hidden_dim"]
        self.pts_intro_out_dim = self.cl_cfg["pts_intro_out_dim"]
        self.img_inter_hidden_dim = self.cl_cfg["img_inter_hidden_dim"]
        self.img_inter_out_dim = self.cl_cfg["img_inter_out_dim"]
        self.pts_inter_hidden_dim = self.cl_cfg["pts_inter_hidden_dim"]
        self.pts_inter_out_dim = self.cl_cfg["pts_inter_out_dim"]

        self.img_feat_dim = self.cl_cfg["img_feat_dim"]
        self.pts_feat_dim = self.cl_cfg["pts_feat_dim"]
        
        self.moco = self.cl_cfg["moco"]
        self.simsiam = self.cl_cfg["simsiam"]
        self.img_moco = self.cl_cfg["img_moco"]
        self.point_intro = self.cl_cfg["point_intro"]
        self.point_branch = self.cl_cfg["point_branch"]

        # self.out_dim = self.cl_cfg["out_dim"]
        # self.hidden_dim = self.cl_cfg["hidden_dim"]
        # self.img_feat_dim = self.cl_cfg["img_feat_dim"]
        # self.pts_feat_dim = self.cl_cfg["pts_feat_dim"]

        if img_backbone:
            self.with_img_backbone = True
            self.img_backbone = builder.build_backbone(img_backbone)
        if pts_backbone:
            self.with_pts_bbox = True
            self.pts_backbone = builder.build_backbone(pts_backbone)

        # self.pts_aggregation = build_sa_module(bbox_head.aggregation_cfg)

        if self.moco:
            self.pts_backbone_k = builder.build_backbone(pts_backbone)
            for param_q, param_k in zip(self.pts_backbone.parameters(), self.pts_backbone_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        if self.img_moco:
            self.img_intro_hidden_dim = self.cl_cfg["img_intro_hidden_dim"]
            self.img_intro_out_dim = self.cl_cfg["img_intro_out_dim"]
            self.img_backbone_k = builder.build_backbone(img_backbone)
            for param_q, param_k in zip(self.img_backbone.parameters(), self.img_backbone_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            self.mlp_intro_img = projection_MLP(
            in_dim=self.img_feat_dim, hidden_dim=self.img_intro_hidden_dim, out_dim=self.img_intro_out_dim)
            self.img_K = self.cl_cfg["img_K"]
        
        # for param_q, param_k in zip(self.pts_aggregation.parameters(), self.pts_aggregation_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        self.mlp_cross_points = projection_MLP(
            in_dim=self.pts_feat_dim, hidden_dim=self.pts_inter_hidden_dim, out_dim=self.pts_inter_out_dim)

        self.mlp_cross_img = projection_MLP(
            in_dim=self.img_feat_dim, hidden_dim=self.img_inter_hidden_dim, out_dim=self.img_inter_out_dim)

        self.mlp_intro_points = projection_MLP(
            in_dim=self.pts_feat_dim, hidden_dim=self.pts_intro_hidden_dim, out_dim=self.pts_intro_out_dim)

        # add a predictor head
        if self.simsiam:
            self.mlp_cross_img_pred = projection_MLP(
                in_dim=self.img_inter_out_dim, hidden_dim=self.img_inter_out_dim, out_dim=self.img_inter_out_dim)
            self.mlp_cross_pts_pred = projection_MLP(
                in_dim=self.pts_inter_out_dim, hidden_dim=self.pts_inter_out_dim, out_dim=self.pts_inter_out_dim)

        self.criterion = NCESoftmaxLoss()
        self.intro_criterion = NCESoftmaxLoss()

        self.img_conv1x1_1 = nn.Conv2d(2048, 2048, 1)
        self.img_conv1x1_2 = nn.Conv2d(2048, 2048, 1)

        # create the queue
        # self.dim = self.cl_cfg["dim"]
        self.K = self.cl_cfg["K"]
        self.m = self.cl_cfg["m"]
        self.T = self.cl_cfg["T"]
    
        if self.moco:
            self.register_buffer("queue", torch.randn(self.K, self.pts_feat_dim))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if self.img_moco:
            self.register_buffer("img_queue", torch.randn(self.img_K, self.img_feat_dim))
            self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
            self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        if self.moco:
            for param_q, param_k in zip(self.pts_backbone.parameters(), self.pts_backbone_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        if self.img_moco:
            for param_q, param_k in zip(self.img_backbone.parameters(), self.img_backbone_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        if self.moco:
            ptr = int(self.queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue[ptr:ptr + batch_size, :] = keys
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[0] = ptr
        
        if self.img_moco:
            img_ptr = int(self.img_queue_ptr)
            assert self.img_K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.img_queue[img_ptr:img_ptr + batch_size, :] = keys
            img_ptr = (img_ptr + batch_size) % self.img_K  # move pointer

            self.img_queue_ptr[0] = img_ptr

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      points_ori=None,
                      img_aug=None
                      ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        loss = {}
        
        if self.point_branch == True:
            _image_features, loc_ori, pts_feat_ori, ori_indices = self.extract_feat(points_ori, img=img, img_metas=img_metas)
            loc_ori = loc_ori[-1] # bs, num_pts, 3
            pts_feat_ori = pts_feat_ori[-1] # bs, feat_dim, num_pts
            
        else:
            _image_features = img_feats = self.extract_img_feat(img, img_metas)

        bs, c , w, h = _image_features.shape
        # image_features = self.img_conv1x1_1(F.interpolate(_image_features, scale_factor=4))
        # image_features = self.img_conv1x1_1(F.interpolate(_image_features, scale_factor=2))
        image_features = _image_features

        ####################################################################
        if self.img_moco == True:
            avg_img_feats_query = self.avgpooling(_image_features).squeeze()
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                img_feats = self.extract_img_feat(img_aug, img_metas)
                avg_img_feats_key = self.avgpooling(img_feats).squeeze()
                
                intro_img_key = self.mlp_intro_img(avg_img_feats_key)
                intro_img_key_normed = F.normalize(intro_img_key, dim=1)

            intro_img_query = self.mlp_intro_img(avg_img_feats_query)
            intro_img_query_normed = F.normalize(intro_img_query,dim=1)

            ## calculate cross loss
            intro_img_logits = torch.mm(
                intro_img_query_normed, intro_img_key_normed.clone().detach().transpose(1, 0))

            intro_img_logits_neg = torch.mm(
                intro_img_query_normed, F.normalize(self.mlp_intro_img(self.img_queue.clone().detach()), dim=1).transpose(1, 0))
            intro_img_logits = torch.cat([intro_img_logits, intro_img_logits_neg], dim=1)

            labels = torch.arange(bs).cuda().long()

            out = torch.div(intro_img_logits, self.T)
            out = out.squeeze().contiguous()

            img_intro_mocov2_loss, cross_acc = self.criterion(out, labels, 
                topk=(int(intro_img_logits.shape[1] / 10), int(intro_img_logits.shape[1] / 2)))

            loss['img_mocov2_loss'] = img_intro_mocov2_loss * self.cl_cfg["img_factor"]

            loss['img_intro_acc_top1'] = cross_acc[0]
            loss['img_intro_acc_top5'] = cross_acc[1]

            self._dequeue_and_enqueue(avg_img_feats_key)


        ####################################################################
        if self.point_branch == True:
            if self.moco == True:
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder()  # update the key encoder

                    points_stack = torch.stack(points, dim=0)
                    x = self.pts_backbone_k(points_stack)
                    loc_t = x['sa_xyz'][-1]
                    pts_feat_t = x['sa_features'][-1]
                    
                    loc_to_ori_list = []
                    for i in range(len(img_metas)):
                        loc_to_ori = apply_3d_transformation(loc_t[i], 'LIDAR', img_metas[i], reverse=True, points_center=self.cl_cfg["points_center"])
                        loc_to_ori_list.append(loc_to_ori)
                    loc_to_ori = torch.stack(loc_to_ori_list, dim=0)

                    # ori(not chage), to_ori(change order downsample_feat_t)
                    intro_cost_dist = torch.cdist(loc_ori, loc_to_ori, p=2).detach().cpu()
                    intro_cost_dist_split = torch.split(intro_cost_dist, 1 ,dim=0)
                    indices = [linear_sum_assignment(intro_cost_dist[i]) for i, c in enumerate(intro_cost_dist_split)]
                    asign = [(torch.as_tensor(i, dtype=torch.int64, device=image_features.device), 
                            torch.as_tensor(j, dtype=torch.int64, device=image_features.device)) for i, j in indices]

                    features_split = torch.split(pts_feat_t, 1 ,dim=0)
                    target_feat_t = [t[0, :, i].permute(1, 0) for t, (_, i) in zip(features_split, asign)]
                    target_feat_t = torch.cat(target_feat_t, dim=0) 
            else:
                points_stack = torch.stack(points, dim=0)
                x = self.pts_backbone(points_stack)
                loc_t = x['sa_xyz'][-1]
                pts_feat_t = x['sa_features'][-1]

                loc_to_ori_list = []
                for i in range(len(img_metas)):
                    loc_to_ori = apply_3d_transformation(loc_t[i], 'LIDAR', img_metas[i], reverse=True, points_center=self.cl_cfg["points_center"])
                    loc_to_ori_list.append(loc_to_ori)
                loc_to_ori = torch.stack(loc_to_ori_list, dim=0)

                # ori(not chage), to_ori(change order downsample_feat_t)
                intro_cost_dist = torch.cdist(loc_ori, loc_to_ori, p=2).detach().cpu()
                intro_cost_dist_split = torch.split(intro_cost_dist, 1 ,dim=0)
                indices = [linear_sum_assignment(intro_cost_dist[i]) for i, c in enumerate(intro_cost_dist_split)]
                asign = [(torch.as_tensor(i, dtype=torch.int64, device=image_features.device), 
                        torch.as_tensor(j, dtype=torch.int64, device=image_features.device)) for i, j in indices]

                features_split = torch.split(pts_feat_t, 1 ,dim=0)
                target_feat_t = [t[0, :, i].permute(1, 0) for t, (_, i) in zip(features_split, asign)]
                target_feat_t = torch.cat(target_feat_t, dim=0) 
            

            # proj
            img_sample_features_list = []
            for i in range(len(img_metas)):
                img_scale_factor = (
                    loc_ori.new_tensor(img_metas[i]['scale_factor'][:2])
                    if 'scale_factor' in img_metas[i].keys() else 1)
                img_flip = img_metas[i]['flip'] if 'flip' in img_metas[i].keys(
                ) else False
                img_crop_offset = (
                    loc_ori.new_tensor(img_metas[i]['img_crop_offset'])
                    if 'img_crop_offset' in img_metas[i].keys() else 0)

                img_sample_features = point_sample(
                    img_metas[i],
                    image_features[i:i+1],
                    loc_ori[i],
                    loc_ori[i].new_tensor(img_metas[i]['lidar2img']),
                    img_scale_factor,
                    img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img_metas[i]['pad_shape'][:2],
                    img_shape=img_metas[i]['img_shape'][:2],
                    aligned=True,
                    padding_mode='zeros',
                    align_corners=True,
                    points_center=self.cl_cfg["points_center"]
                )

                img_sample_features_list.append(img_sample_features)
            
            img_sample_features = torch.stack(img_sample_features_list, dim=0)
            bs, N, C = img_sample_features.shape
            pts_feat_ori = pts_feat_ori.permute(0, 2, 1)
            pts_feat_ori = pts_feat_ori.reshape(-1, 1024)
            img_sample_features = img_sample_features.reshape(-1, 2048)

            pts_feat_ori_intro_normed = F.normalize(self.mlp_intro_points(pts_feat_ori),dim=1)
            target_features_normed = F.normalize(self.mlp_intro_points(target_feat_t), dim=1)

            if self.simsiam == True:
                p_pts = self.mlp_cross_points(pts_feat_ori)
                p_img = self.mlp_cross_img(img_sample_features)

                z_pts = self.mlp_cross_pts_pred(p_pts)
                z_img = self.mlp_cross_img_pred(p_img)
                
                # cross_pointcontrastive = D(p_img, z_pts, version='original')

                # without cut grad
                cross_pointcontrastive = D(p_img, z_pts, version='original')/2 + D(p_pts, z_img, version='original')/2

                # calculate acc for log
                logits = torch.mm(
                    p_img, p_pts.clone().detach().transpose(1, 0))

                labels = torch.arange(bs * N).cuda().long()
                out = torch.div(logits, self.T)
                out = out.squeeze().contiguous()

                # top 10% and top 50%
                _, cross_acc = self.criterion(out, labels, topk=(int(p_img.shape[0] / 10), int(p_img.shape[0] / 2)))

                loss['cross_acc_top1'] = cross_acc[0]
                loss['cross_acc_top5'] = cross_acc[1]

            
                
            else:
                # pts_feat_ori_cross_normed = F.normalize(self.mlp_cross_points(target_feat_t),dim=1)
                pts_feat_ori_cross_normed = F.normalize(self.mlp_cross_points(pts_feat_ori),dim=1)
                sample_img_feat_normed = F.normalize(self.mlp_cross_img(img_sample_features),dim=1)

                ## calculate cross loss
                logits = torch.mm(
                    sample_img_feat_normed, pts_feat_ori_cross_normed.clone().detach().transpose(1, 0))

                ## not cut loss
                # logits = torch.mm(
                #     sample_img_feat_normed, pts_feat_ori_cross_normed.transpose(1, 0))

                # if self.moco == True:  
                #     logits_neg = torch.mm(
                #         sample_img_feat_normed, F.normalize(self.mlp_cross_points(self.queue.clone().detach()), dim=1).transpose(1, 0))
                #     logits = torch.cat([logits, logits_neg], dim=1)

                labels = torch.arange(bs * N).cuda().long()
                out = torch.div(logits, self.T)
                out = out.squeeze().contiguous()

                cross_pointcontrastive, cross_acc = self.criterion(out, labels, 
                    topk=(int(pts_feat_ori_cross_normed.shape[0] / 10), int(pts_feat_ori_cross_normed.shape[0] / 2)))

                loss['cross_acc_top1'] = cross_acc[0]
                loss['cross_acc_top5'] = cross_acc[1]

                ##### for visual
                # if  torch.distributed.get_rank() == 0:
                #     print("yes")
                #     savepoint = {}
                #     savepoint["img"] = img
                    
                #     savepoint["rect"] = img_metas[0]["rect"]
                #     savepoint["Trv2c"] = img_metas[0]["Trv2c"]
                #     savepoint["P2"] = img_metas[0]["P2"]

                #     savepoint["points_transed"] = points # transed
                #     savepoint["points_ori"] = points_ori # ori location

                #     savepoint["sample_transed_points_location"] = downsample_loc_t # sampled transed location
                #     savepoint["sample_points_location_ori"] = downsample_loc_ori # sampled ori location

                #     # savepoint["vote_points1_feats"] = intro_pts_feat_normed
                #     # savepoint["vote_points2_feats"] = target_features_normed

                #     savepoint["sample_points_to_ori"] = loc_to_ori # sampled to ori

                #     transed_scene_points = apply_3d_transformation(points[0][:, :3], 'LIDAR', img_metas[0], reverse=True, points_center=self.cl_cfg["points_center"])
                #     savepoint["points_to_ori"] = transed_scene_points # sampled to ori
                    
                #     torch.save(savepoint, "savepoint.pth")
                #     exit(-100000)
                # exit(-100000)

            ## calculate intro loss
            intro_logits = torch.mm(
                pts_feat_ori_intro_normed, target_features_normed.clone().detach().transpose(1, 0))

            if self.moco == True:  
                intro_logits_neg = torch.mm(
                    pts_feat_ori_intro_normed, F.normalize(self.mlp_intro_points(self.queue.clone().detach()), dim=1).transpose(1, 0))
                intro_logits = torch.cat([intro_logits, intro_logits_neg], dim=1)

            intro_labels = torch.arange(bs * N).cuda().long()
            intro_out = torch.div(intro_logits, self.T)
            intro_out = intro_out.squeeze().contiguous()

            intro_pointcontrastive, intro_acc = self.intro_criterion(intro_out, intro_labels)

            loss['cross_loss'] = cross_pointcontrastive * self.cl_cfg["cross_factor"]

            if self.point_intro == True:
                loss['intro_loss'] = intro_pointcontrastive
                loss['intro_acc_top1'] = intro_acc[0]
                loss['intro_acc_top5'] = intro_acc[1]
            

            if self.moco == True:  
                self._dequeue_and_enqueue(pts_feat_ori)
        
        return loss

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None

        img_contrastive = img_feats[-1]
        return img_contrastive

    def extract_pts_feat(self, points, img_metas):
        """Extract features of points."""
        points_stack = torch.stack(points, dim=0)
        x = self.pts_backbone(points_stack)
        return x['sa_xyz'], x['sa_features'], x['sa_indices']


    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[batch_id]['boxes_3d']

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        xyz, pts_feats, indices = self.extract_pts_feat(points, img_metas)
        return (img_feats, xyz, pts_feats, indices)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]

        return bbox_list


def point_sample(
    img_meta,
    img_features,
    points,
    lidar2img_rt,
    img_scale_factor,
    img_crop_offset,
    img_flip,
    img_pad_shape,
    img_shape,
    aligned=True,
    padding_mode='zeros',
    align_corners=True,
    points_center=None,
):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        lidar2img_rt (torch.Tensor): 4x4 transformation matrix.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    # points = apply_3d_transformation(points, 'LIDAR', img_meta, reverse=True, points_center=points_center)

    # project points from velo coordinate to camera coordinate
    num_points = points.shape[0]
    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
    pts_2d = pts_4d @ lidar2img_rt.t()

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate

    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor

    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset
    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1

    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    return point_features.squeeze().t()
