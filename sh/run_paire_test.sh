#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python /home/skl/yl/kge_tool/run_sys_test.py --train --cuda --max_step 150001 --dim $2 --lr $3 --decay $4 --save_path $5 \
--gamma  $6 \
--level $7  $8  ${9} ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23}

