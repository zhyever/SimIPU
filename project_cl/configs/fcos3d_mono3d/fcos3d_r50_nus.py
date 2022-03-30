_base_ = [
    '../../config/_base_/datasets/nus-mono3d.py', '../../config/_base_/models/fcos3d.py',
    '../../config/_base_/schedules/mmdet_schedule_1x.py', '../../config/_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        # norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=False,
        # norm_eval=True,
        style='pytorch',
        # style='caffe'
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)
    ))

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.008, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    # warmup_iters=1500, # for moco
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12
evaluation = dict(interval=2)

# load_from=None
# load_from='checkpoints/waymo_ep50_with_backbone.pth'
# load_from='checkpoints/imgsup_finetune_waymo_ep1_with_backbone.pth'
# load_from='checkpoints/resnet50-19c8e357_convert_mono3d.pth'
# load_from='checkpoints/imgsup_finetune_waymo_ep5_with_backbone_repro.pth'
# load_from='checkpoints/imgsup_finetune_waymo_ep5_with_backbone_moco.pth'
# load_from=None
# load_from='checkpoints/mono3d_waymo_half.pth'
# load_from='checkpoints/mono3d_waymo_oneten.pth'
load_from='checkpoints/mono3d_waymo_full.pth'
# load_from='checkpoints/mono3d_waymo_onefive.pth'
