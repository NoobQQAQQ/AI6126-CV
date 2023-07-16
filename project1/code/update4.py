# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
gpu_ids = range(1)
seed = 2101985

model = dict(
    type='EncoderDecoder',
    # pretrained='resnet50_v1c.pth',
    # pretrained='swin_base_patch4_window7_224.pth',  # 22k imagenet
    # # pretrained=None,
    # backbone=dict(
    #     type='SwinTransformer',
    #     pretrain_img_size=224,
    #     embed_dims=128,  # 大
    #     patch_size=4,
    #     window_size=7,
    #     mlp_ratio=4,
    #     depths=[2, 2, 18, 2],  # 大
    #     num_heads=[4, 8, 16, 32],  # 大
    #     strides=(4, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     qkv_bias=True,
    #     qk_scale=None,
    #     patch_norm=True,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.3,
    #     use_abs_pos_embed=False,
    #     act_cfg=dict(type='GELU'),
    #     norm_cfg=dict(type='LN', requires_grad=True)
    # ),
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
        # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
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
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                        dict(type='LovaszLoss', reduction='none', loss_weight=0.2)]
        # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        ),
    # model training and testing settings
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
# load_from = '../result/update4/epoch_100.pth'
resume_from = None
# resume_from = '../result/update4/epoch_100.pth'
workflow = [('train', 1)]
cudnn_benchmark = True

# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.002)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)


log_config = dict(
    interval=1000, hooks=[dict(type='TextLoggerHook', by_epoch=True)]) # 其实还是按iteration来打log的
runner = dict(type='EpochBasedRunner', max_epochs=30) #按epoch的方式进行迭代
checkpoint_config = dict(by_epoch=True, interval=30) #每多少epoch保存一次模型
evaluation = dict(interval=3, metric='mIoU', pre_eval=True)  # 每多少epoch计算一次指标

# runner = dict(type='IterBasedRunner', max_iters=20000)
# checkpoint_config = dict(by_epoch=False, interval=10000)
# # evaluation = dict(interval=2000, metric=['mIoU', 'mDice'], pre_eval=True)
# evaluation = dict(interval=500, metric='mIoU', pre_eval=True)