# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
gpu_ids = range(1)
seed = 2101985

model = dict(
    type='EncoderDecoder',
    # pretrained='resnet50_v1c.pth',
    # backbone=dict(
    #     type='ResNetV1c',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     dilations=(1, 1, 2, 4),
    #     strides=(1, 2, 1, 1),
    #     norm_cfg=norm_cfg,  # batch norm config
    #     norm_eval=False,
    #     style='pytorch',
    #     contract_dilation=True),
    pretrained='resnest50.pth',
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,  # batch norm config
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),    
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,  # batch norm config 
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                        dict(type='LovaszLoss', reduction='none', loss_weight=0.5)]
        ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,  # batch norm config
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                        dict(type='LovaszLoss', reduction='none', loss_weight=0.2)]
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

dataset_type = 'CelebAMaskDataset'
data_root = '../data/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train1/train_image',
        ann_dir='train1/train_mask',
        # split='mini_train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/val_image',
        ann_dir='val/val_mask',
        # split='mini_val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/test_image',
        ann_dir='test/test_mask',
        # split='mini_test.txt',
        pipeline=test_pipeline))

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from = '../result/update2/epoch_100.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.003)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=0.00001, by_epoch=False)

log_config = dict(
    interval=1000, hooks=[dict(type='TextLoggerHook', by_epoch=True)]) # 其实还是按iteration来打log的
runner = dict(type='EpochBasedRunner', max_epochs=30) #按epoch的方式进行迭代
checkpoint_config = dict(by_epoch=True, interval=30) #每多少epoch保存一次模型
evaluation = dict(interval=3, metric='mIoU', pre_eval=True)  # 每多少epoch计算一次指标