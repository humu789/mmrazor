_base_ = ['./detnas_frcnn_shufflenet_supernet_coco_1x.py']

_base_.val_dataloader.batch_size = 1
_base_.val_dataloader.num_workers = 1

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=2,
    num_candidates=5,
    top_k=2,
    num_mutation=2,
    num_crossover=2,
    mutate_prob=0.1,
    flops_constraint=('backbone', (0., 330 * 1e6)),
    score_key='bbox_mAP')