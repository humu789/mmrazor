_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

_base_.val_dataloader.batch_size = 512
_base_.val_dataloader.num_workers = 8

# TODO: to del it after mmengine's bug fixed
_base_.param_scheduler.convert_to_iter_based = False

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=20,
    num_candidates=50,
    top_k=10,
    num_mutation=25,
    num_crossover=25,
    mutate_prob=0.1
)
