_base_ = ['mmdet::retinanet/retinanet_r50_fpn_1x_coco.py']

retina = _base_.model
# data_preprocessor = retina.data_preprocessor
float_ckpt = '/mnt/petrelfs/caoweihan.p/ckpt/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'  # noqa: E501

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQPerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.LSQMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
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
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    architecture=retina,
    # float_checkpoint=float_ckpt,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.base_dense_head.BaseDenseHead.predict_by_feat',
                'mmdet.models.detectors.base.BaseDetector.add_pred_to_datasample',
                'mmdet.models.dense_heads.anchor_head.AnchorHead.loss_by_feat',
                # 'mmcls.models.heads.ClsHead._get_loss',
                # 'mmcls.models.heads.ClsHead._get_predictions'
            ])))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=12,
    by_epoch=True,
    begin=0,
    end=12)

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=12,
    val_interval=1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')
# test_cfg = val_cfg

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=-1,
        out_dir='/mnt/petrelfs/caoweihan.p/training_ckpt/lsq')
)
