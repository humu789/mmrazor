import torch
from mmcv import Config

from mmrazor.models import build_algorithm

pth_path = 'ckpt/best.pth.tar'
save_path = 'ckpt/best_new_ema.pth.tar'
config_path = 'configs/nas/greedynas/greedynas_subnet_mobilenet_8xb96_in1k.py'
subnet_path = 'configs/nas/greedynas/final_subnet_op_paper.yaml'
model = 'ema_model'

def main():
    keys_ckpt_all = []
    keys_ckpt_mapping = []
    keys_model = []

    ckpt = torch.load(pth_path, map_location='cpu')
    for k, v in ckpt[model].items():
        keys_ckpt_all.append(k)
        if k.split('.')[-1] not in [
                'running_mean', 'running_var', 'num_batches_tracked'
        ]:
            keys_ckpt_mapping.append(k)

    print('#' * 100)

    cfg = Config.fromfile(config_path)
    cfg_options = dict()
    cfg_options['algorithm.mutable_cfg'] = subnet_path
    cfg.merge_from_dict(cfg_options)
    algorithm = build_algorithm(cfg.algorithm)
    # model = algorithm.architecture
    for name, param in algorithm.named_parameters():
        keys_model.append(name)

    assert len(keys_ckpt_mapping) == len(keys_model)

    mapping = dict()
    for i in range(len(keys_model)):
        key = keys_ckpt_mapping[i][:keys_ckpt_mapping[i].rfind('.')]
        val = keys_model[i][:keys_model[i].rfind('.')]
        mapping[key] = val

    for k, v in mapping.items():
        print(f'{k}\t{v}')

    print(f'mapping size: {len(mapping)}')

    for k in keys_ckpt_all:
        old_name = k[:k.rfind('.')]
        new_name = mapping[old_name]
        new_k = k.replace(old_name, new_name)
        ckpt[model][new_k] = ckpt[model].pop(k)

    for k, v in ckpt[model].items():
        print(f'{k}\t\t{v.size()}')

    torch.save(ckpt[model], save_path)

def check():
    ckpt_org = torch.load(pth_path, map_location='cpu')
    # ckpt_new = torch.load(save_path, map_location='cpu')
    print(ckpt_org.keys())

    # print(list(ckpt_org.values())[0])
    # print('#'*100)
    # print(list(ckpt_new.values())[0])

if __name__ == '__main__':
    main()
    # check()