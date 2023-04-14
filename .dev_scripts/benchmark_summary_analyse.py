import argparse
import os

import mmengine


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyse summary.yml generated by benchmark test')
    parser.add_argument('file_path', help='Summary.yml path')
    args = parser.parse_args()
    return args


metric_mapping = {
    'Top 1 Accuracy': 'accuracy/top1',
    'Top 5 Accuracy': 'accuracy/top5',
    'box AP': 'coco/bbox_mAP'
}


def compare_metric(result, metric):
    expect_val = result['expect'][metric]
    actual_val = result['actual'].get(metric_mapping[metric], None)
    if actual_val is None:
        return None, None
    if metric == 'box AP':
        actual_val *= 100
    decimal_bit = len(str(expect_val).split('.')[-1])
    actual_val = round(actual_val, decimal_bit)
    error = round(actual_val - expect_val, decimal_bit)
    error_percent = round(abs(error) * 100 / expect_val, 3)
    return error, error_percent


def main():
    args = parse_args()
    file_path = args.file_path
    results = mmengine.load(file_path, 'yml')
    miss_models = dict()
    sort_by_error = dict()
    for k, v in results.items():
        valid_keys = v['expect'].keys()
        compare_res = dict()
        for m in valid_keys:
            error, error_percent = compare_metric(v, m)
            if error is None:
                continue
            compare_res[m] = {'error': error, 'error_percent': error_percent}
            if error != 0:
                miss_models[k] = compare_res
                sort_by_error[k] = error
    sort_by_error = sorted(
        sort_by_error.items(), key=lambda x: abs(x[1]), reverse=True)
    miss_models_sort = dict()
    miss_models_sort['total error models'] = len(sort_by_error)
    for k_v in sort_by_error:
        index = k_v[0]
        miss_models_sort[index] = miss_models[index]
    save_path = os.path.join(os.path.dirname(file_path), 'summary_error.yml')
    mmengine.fileio.dump(miss_models_sort, save_path, sort_keys=False)
    print(f'Summary analysis result saved in {save_path}')


if __name__ == '__main__':
    main()
