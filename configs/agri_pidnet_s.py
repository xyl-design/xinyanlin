default_scope = 'mmseg'
custom_imports = dict(imports=['agri_pidnet'], allow_failed_imports=False)

crop_size = (1024, 1024)
base_scale = (2048, 2048)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    type='AgriEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(
        type='AgriPIDNetSHead',
        m=2,
        n=3,
        num_classes=4,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        use_ca_mkir=True,
        use_sae_d2t=True,
        use_bwr_bag=True,
        use_ca=True,
        use_mbcr=True,
        use_window_attention=True,
        use_boundary_loss=True,
        boundary_loss_weight=0.20,
        edge_width=4,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=16384),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.40,
                class_weight=[1.0, 1.0, 1.0, 5.0],
                avg_non_ignore=True,
                loss_name='loss_ce'),
            dict(
                type='AgriDiceLoss',
                use_sigmoid=False,
                loss_weight=0.25,
                class_weight=[0.5, 0.5, 0.5, 3.0],
                loss_name='loss_dice'),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                classes='all',
                per_image=False,
                reduction='none',
                loss_weight=0.15,
                class_weight=[1.0, 1.0, 1.0, 4.0],
                loss_name='loss_lovasz'),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=base_scale,
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

dataset_type = 'CottonFieldDataset'
data_root = 'data/cotton_two_stage'

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/masks'),
        pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images', seg_map_path='val/masks'),
        pipeline=test_pipeline,
        test_mode=True),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/masks'),
        pipeline=test_pipeline,
        test_mode=True),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        by_epoch=False,
        begin=1500,
        end=20000),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
work_dir = 'work_dirs/agri_pidnet_s'
randomness = dict(seed=0, deterministic=True)
