


CUDA_VISIBLE_DEVICES=0 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 12 --learning_rate 0.005 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0324_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 12 --learning_rate 0.0005 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0324_02 &
CUDA_VISIBLE_DEVICES=2 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 5 --learning_rate 0.005 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0324_03 &
CUDA_VISIBLE_DEVICES=3 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 5 --learning_rate 0.0005 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0324_04 &



CUDA_VISIBLE_DEVICES=0 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 15 --learning_rate 0.0005 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0325_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 15 --learning_rate 0.0001 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0325_02 &
CUDA_VISIBLE_DEVICES=2 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 8 --learning_rate 0.0005 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0325_03 &
CUDA_VISIBLE_DEVICES=3 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 8 --learning_rate 0.0001 --mode RotatE --save_path /home/skl/yl/ce_project/relation_cl/models/relation/rotate_relion_0325_04 &



CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.00005 --gamma 8 --learning_rate 0.0001 --mode ComplEx --save_path /home/skl/yl/ce_project/relation_cl/models/relation/ComplEx_attention_relion_0325_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.0001 --mode ComplEx --save_path /home/skl/yl/ce_project/relation_cl/models/relation/ComplEx_attention_relion_0325_02 &


CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.00005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/skl/yl/ce_project/relation_cl/models/relation/ComplEx_attention_relion_0325_03 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/skl/yl/ce_project/relation_cl/models/relation/ComplEx_attention_relion_0325_04 &

CUDA_VISIBLE_DEVICES=2 python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/skl/yl/ce_project/relation_cl/models/relation/ComplEx_attention_relion_0325_05
