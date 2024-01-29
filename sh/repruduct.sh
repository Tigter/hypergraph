#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python /home/skl/yl/kge_tool/repreuduct_mode.py --train --cuda --max_step 100001 --dim $2 --lr $3 --decay $4 --save_path $5 \
--gamma  $6 \
--level $7  $8  ${9} ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23}
# CUDA_VISIBLE_DEVICES=$1 python train_ons.py --cuda --max_step 200000 --on_dim $2 --lr $3 --decay $4 --save_path $5 --gamma_m $6 \
# --beta  $7 --alpha $8 --gamma $9 ${10}
