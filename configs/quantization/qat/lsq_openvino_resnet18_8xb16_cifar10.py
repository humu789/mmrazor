_base_ = ['mmcls::resnet/resnet18_8xb16_cifar10.py']

resnet = _base_.model
float_ckpt = '/mnt/petrelfs/caoweihan.p/ckpt/resnet18_b16x8_cifar10_20210528-bd6371c8.pth'  # noqa: E501

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQPerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.LSQMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    # w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    # a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    # w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    # a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True),
)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=10,
        # RGB format normalization parameters
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        # loaded images are already RGB format
        to_rgb=False),
    architecture=resnet,
    # float_checkpoint=float_ckpt,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=100,
    by_epoch=True,
    begin=0,
    end=100)

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=100,
    val_interval=1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')
# test_cfg = val_cfg

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=-1,
        out_dir='/mnt/petrelfs/caoweihan.p/training_ckpt/lsq')
)
