################################# FINAL SETTING! 
# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# MOCO Model
model = dict(
    # type='Inter_Intro_moco',
    type='Inter_Intro_moco_better',

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        # norm_cfg=dict(type='BN'), # for debug
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        norm_eval=False,
        style='pytorch'),

    # With MOCO
    pts_backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 1024, (512, 512)),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((32, 32, 64), (32, 32, 64), (16, 16, 16)),
        sa_channels=(((32, 32, 64), (32, 32, 64), (64, 64, 128)),
                     ((64, 64, 128), (64, 64, 128), (128, 128, 256)),
                     ((128, 128, 256), (128, 128, 256), (256, 256, 512))),
        aggregation_channels=(128, 256, 1024),
        fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=((-1), (-1), (512, -1)),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            # use_xyz=True,
            use_xyz=False,
            normalize_xyz=False)),

    # model training and testing settings
    train_cfg=dict(
        cl_strategy = dict(
            pts_intro_hidden_dim=1024,
            pts_intro_out_dim=128,
            img_inter_hidden_dim=2048,
            img_inter_out_dim=128,
            pts_inter_hidden_dim=1024,
            pts_inter_out_dim=128,
            pts_feat_dim=1024,
            img_feat_dim=2048,
            K=8192*4,
            m=0.999,
            T=0.07,
            points_center=[35.2, 0, -1],
            cross_factor=1,
            moco=False,
            simsiam=False,
            ############################################
            img_moco=False,
            point_intro=True, # intro-loss
            point_branch=True  # if pts backbone
            )))

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_modality = dict(use_lidar=True, use_camera=True)
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
#     classes=class_names,
#     sample_groups=dict(Car=15))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range), # filter range
    dict(type='IndoorPointSample', num_points=16384), # sample here only for pretrain!
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),

    ############################## 
    dict(
        type='Resize',
        # img_scale=[(640, 192), (2560, 768)],
        img_scale=[(640, 192), (2400, 720)],
        multiscale_mode='range',
        keep_ratio=True),
    ##############################

    dict(
        type='GlobalRotScaleTrans',
        # rot_range=[-0.78539816, 0.78539816],
        # scale_ratio_range=[0.95, 1.05],
        rot_range=[-1.5707963, 1.5707963],
        scale_ratio_range=[0.75, 1.25],
        translation_std=[0, 0, 0],
        points_center=[35.2, 0, -1]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'points_ori']),
]

test_pipeline = [] # No need to test

# for dataset
pretraining=True
cross=True # for cross pretrain
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    # samples_per_gpu=3,
    # workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            pretraining=True,
            cross=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    
    # actually there is no val
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        pretraining=True,
        box_type_3d='LiDAR'))

# Not be used in pretrain
evaluation = dict(start=9999, interval=1) # No use


# optimizer
optimizer = dict(
    constructor='HybridOptimizerConstructor',
    pts=dict(
        type='AdamW',
        # lr=0.002,
        lr=0.001,
        betas=(0.95, 0.99),
        weight_decay=0.01,
        step_interval=1),
    img=dict(
        type='SGD',
        # lr=0.03,
        lr=0.03,
        momentum=0.9,
        weight_decay=0.0001,
        step_interval=1),
    mlp=dict(
        type='SGD',
        # lr=0.03,
        lr=0.03,
        momentum=0.9,
        weight_decay=0.0001,
        step_interval=1))

 
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup='linear', warmup_iters=10, warmup_ratio=0.001, warmup_by_epoch=True)
lr_config = dict(policy='Exp', gamma=0.99)

# runtime settings
checkpoint_config = dict(interval=5)

# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

total_epochs = 100

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

find_unused_parameters=True # I cannot find it