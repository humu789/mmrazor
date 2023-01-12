_base_ = ['mmdet::rtmdet/rtmdet_x_8xb32-300e_coco.py']

test_pipeline = _base_.test_pipeline
train_dataloader = dict(
    batch_size=32,
    dataset=dict(pipeline=test_pipeline))

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=train_dataloader,
    calibrate_steps=32,
)

retina = _base_.model
# data_preprocessor = retina.data_preprocessor
float_ckpt = '/mnt/petrelfs/caoweihan.p/ckpt/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'  # noqa: E501

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
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
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    architecture=retina,
    float_checkpoint=float_ckpt,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.rtmdet_head.RTMDetHead.predict_by_feat',
                'mmdet.models.dense_heads.rtmdet_head.RTMDetHead.loss_by_feat',
            ])))

# model_wrapper_cfg = dict(
#     type='mmrazor.MMArchitectureQuantDDP',
#     broadcast_buffers=False,
#     find_unused_parameters=True)

custom_hooks = []
