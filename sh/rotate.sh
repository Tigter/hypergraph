


CUDA_VISIBLE_DEVICES=0 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 12 --learning_rate 0.005 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0324_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 12 --learning_rate 0.0005 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0324_02 &
CUDA_VISIBLE_DEVICES=2 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 5 --learning_rate 0.005 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0324_03 &
CUDA_VISIBLE_DEVICES=3 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 5 --learning_rate 0.0005 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0324_04 &



CUDA_VISIBLE_DEVICES=0 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 15 --learning_rate 0.0005 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0325_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 15 --learning_rate 0.0001 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0325_02 &
CUDA_VISIBLE_DEVICES=2 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 8 --learning_rate 0.0005 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0325_03 &
CUDA_VISIBLE_DEVICES=3 nohup python relation_predict.py --cuda --train --test --batch_size 2048 --gamma 8 --learning_rate 0.0001 --mode RotatE --save_path /home/tengwei/hypergraph/models/models/relation/rotate_relion_0325_04 &



CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.00005 --gamma 8 --learning_rate 0.0001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_relion_0325_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.0001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_relion_0325_02 &


CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.00005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_relion_0325_03 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_relion_0325_04 &

CUDA_VISIBLE_DEVICES=2 python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_relion_0325_05


CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_all_relion_0325_01 &
CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 200 --learning_rate 0.005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_all_relion_0325_02 &
CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_all_relion_0325_03 &
CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 200 --learning_rate 0.001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_all_relion_0325_04 &



CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_only_relion_0325_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 200 --learning_rate 0.005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_only_relion_0325_02 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_only_relion_0325_03 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 200 --learning_rate 0.001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_only_relion_0325_04 &


CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.0001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_only_relion_0325_05 &

CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 200 --learning_rate 0.0001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_only_relion_0325_07 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 200 --learning_rate 0.0005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/ComplEx_attention_only_relion_0325_08 &


CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.0005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/Filtered_data_ComplEx_attention_only_relion_0330_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0005 --gamma 8 --learning_rate 0.0005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/Filtered_data_DistMult_attention_only_relion_0330_01 &

CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0000 --gamma 8 --learning_rate 0.0005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/Filtered_data_RotatE_attention_only_relion_0330_01 &
CUDA_VISIBLE_DEVICES=1 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0000 --gamma 8 --learning_rate 0.0001 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/Filtered_data_RotatE_attention_only_relion_0330_02 &
CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0000 --gamma 16 --learning_rate 0.0005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/Filtered_data_RotatE_attention_only_relion_0330_03 &
CUDA_VISIBLE_DEVICES=0 nohup python relation_predict_v2.py --cuda --train --test --batch_size 2048 --regularization 0.0000 --gamma 24 --learning_rate 0.0005 --mode ComplEx --save_path /home/tengwei/hypergraph/models/models/relation/Filtered_data_RotatE_attention_only_relion_0330_04 &



