

import random
from .calib_utils import Calibration, get_lidar_in_image_fov, draw_lidar, show_lidar_on_image
import torch
import cv2
from PIL import Image
import time

class Frustum_Region(object):
    def __init__(self, wr_max=0.7, wr_min=0.4, hr_max=1.0, hr_min=0.4, th=0.7, p=0.5):
        super(Frustum_Region, self).__init__()
        self.wr_max=wr_max
        self.wr_min=wr_min
        self.hr_max=hr_max
        self.hr_min=hr_min
        self.th=th
        self.p=p

    def __call__(self, input_dict):
        # if random.random()>self.p: # p=0.5 more opportunities to global views
        #     return input_dict

        idx =input_dict['sample_idx']
        ori_h,ori_w = (input_dict['ori_shape'][:2]
            if 'ori_shape' in input_dict.keys() else 1)
        img = input_dict['img']
        points = input_dict['points'].tensor
        calib = Calibration(input_dict['P2'],input_dict['Trv2c'],input_dict['rect'])
        img_scale_factor = (
            input_dict['scale_factor'][:2]
            if 'scale_factor' in input_dict.keys() else 1)
        print("Check", points.shape)
        img_scale_factor = (1)

        # random select 2d region
        h, w = img.shape[:2]
        region_w = int(random.uniform(self.wr_min, self.wr_max)* w)
        region_h = int(random.uniform(self.hr_min, self.hr_max)* h)
        x1 = random.randint(0, w-region_w)
        y1 = random.randint(max(0, int(self.th*h-region_h)), h-region_h) # mainly focus on bottom regions
        x2,y2 = x1+region_w, y1+region_h

        # get Frustum 3D Region
        # option1
        points = points.numpy()
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(points[:,0:3],
            calib, 0, 0, ori_w, ori_h, True)
        # # option2
        # # project points from velo coordinate to camera coordinate
        # num_points = points.shape[0]
        # pts_4d = torch.cat([points[:,0:3], points.new_ones(size=(num_points, 1))], dim=-1)
        # pts_2d = pts_4d @ torch.tensor(input_dict['lidar2img']).t()
        # # cam_points is Tensor of Nx4 whose last column is 1
        # # transform camera coordinate to image coordinate
        # pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
        # pts_2d[:, 0] /= pts_2d[:, 2] # normalization
        # pts_2d[:, 1] /= pts_2d[:, 2] # normalization
        # pc_image_coord = pts_2d

        # Filter
        pc_image_coord = pc_image_coord[:, 0:2] * img_scale_factor
        box_fov_inds = (pc_image_coord[:,0]<x2) & \
                    (pc_image_coord[:,0]>=x1) & \
                    (pc_image_coord[:,1]<y2) & \
                    (pc_image_coord[:,1]>=y1)
        pc_in_box_fov = points[box_fov_inds]
        
        input_dict['img'] = img[y1:y2,x1:x2]
        input_dict['points'].tensor = torch.tensor(pc_in_box_fov)

        print("P", points[:3,:])
        # visualize
        # draw_lidar(points, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None)
        # show_lidar_on_image(pc_in_box_fov[:,0:3], img.copy(), calib, ori_w, ori_h, img_scale_factor, 'lidar_in_region_{}'.format(idx),  region=(x1,y1,x2,y2))
        show_lidar_on_image(points[:,0:3], img.copy(), calib, ori_w, ori_h, img_scale_factor, 'lidar_in_image_{}'.format(idx))
        # time.sleep(30)
        # print(hhh)
        return input_dict