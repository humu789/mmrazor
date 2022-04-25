#!/usr/bin/env bash

set -x

CONFIG="configs/nas/greedynas/greedynas_subnet_mobilenet_8xb96_in1k.py"
SAVE_PATH="tools/visualizations/results/lr.png"

python -u $(dirname "$0")/vis_lr.py ${CONFIG} --save-path=${SAVE_PATH} \
  --ngpus=4 \
