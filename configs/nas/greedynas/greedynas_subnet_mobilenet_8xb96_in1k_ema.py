_base_ = [
    '../../_base_/schedules/mmcls/imagenet_bs768_autoaug_greedynas.py',
    '../../_base_/mmcls_runtime.py',
    '../../_base_/datasets/mmcls/imagenet_bs96_pil_resize_autoaug_greedynas.py'
]

init_cfg = [
    dict(type='Kaiming', layer=['Conv2d']),
    dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm']),
    dict(type='Kaiming', layer=['Linear'],distribution='uniform', nonlinearity='linear')
]
norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    init_cfg=init_cfg,
    backbone=dict(
        type='SearchableMobileNet',
        arch_setting_type='greedynas',
        widen_factor=1.0,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearDropoutClsHead',
        num_classes=1000,
        in_channels=1280,
        dropout_rate=0.2,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

act_hswish = dict(type='HSwish', inplace=True)
act_hsigmoid = dict(
    type='HSigmoid', bias=3, divisor=6, min_value=0, max_value=1)

se_cfg = dict(ratio=4, divisor=1, act_cfg=(act_hswish, act_hsigmoid))

mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        searchable_blocks=dict(
            type='OneShotOP',
            choices=dict(
                mb_k3e3=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k5e3=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k7e3=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k3e6=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k5e6=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k7e6=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k3e3_se=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k5e3_se=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k7e3_se=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k3e6_se=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k5e6_se=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                mb_k7e6_se=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish),
                identity=dict(type='Identity'))),
        first_blocks=dict(
            type='OneShotOP',
            choices=dict(
                mb_k3e1=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_hswish)))))

algorithm = dict(
    type='GreedyNAS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    mutable_cfg='configs/nas/greedynas/final_subnet_op_paper.yaml',
    distiller=None,
    retraining=True,
)

workflow = [('train', 1)]
evaluation = dict(interval=1, metric='accuracy', save_best='accuracy_top-1')

# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

custom_hooks = [dict(type='EMAHook', momentum=0.0001,priority=49)]

find_unused_parameters = False
