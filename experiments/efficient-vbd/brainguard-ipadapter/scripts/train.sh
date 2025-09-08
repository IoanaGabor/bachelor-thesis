# This file is an adaptation of https://github.com/kunzhan/BrainGuard/blob/main/scripts/train.sh

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

. $CONDA_ROOT/etc/profile.d/conda.sh

conda activate neuraldiffuser


python -c "import os, torch; print(torch.cuda.device_count() if 'CUDA_VISIBLE_DEVICES' not in os.environ else len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))"

now=$(date +"%Y%m%d_%H%M%S")

train_type=vision
save_path=./logs/$train_type/
data_root=./data/natural-scenes-dataset

python -u main.py \
-ls 1 \
-gr 600 \
--cuda_id '{"server":-1, "1": 0, "2": 1, "5": 2}' \
-tp $train_type \
-p 24 \
-lbs 50 \
--data_root $data_root \
2>&1 | tee $save_path$now.log
