CUDA_VISIBLE_DEVICES=0 python hyper_graph.py --debug --configName complex_01 --save_path ./models/HyperGraph_config_1 --init ./models/models//HyperGraph_config_1/MRR 

CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph.py --debug --configName complex_02 --save_path ./models/HyperGraph_config_2 --init ./models/models//HyperGraph_config_2/MRR  &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph.py --debug --configName complex_03 --save_path ./models/HyperGraph_config_3 --init ./models/models//HyperGraph_config_3/MRR &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph.py --debug --configName complex_04 --save_path ./models/HyperGraph_config_4 --init ./models/models//HyperGraph_config_4/MRR &


CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph.py --train --configName complex_ind_02 --save_path ./models/HyperGraph_inductive_2 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph.py --train --configName complex_ind_01 --save_path ./models/HyperGraph_inductive_1 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph.py --train --configName complex_ind_03 --save_path ./models/HyperGraph_inductive_3 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph.py --train --configName complex_ind_04 --save_path ./models/HyperGraph_inductive_4 &

CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_sampler.py --train --configName inductive_base_01 --save_path ./models/HyperGraph_inductive_new > 237_inductive_debug.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_sampler.py --train --configName 237_base_01 --save_path ./models/HyperGraph_237_new > 237_debug.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_sampler.py --train --configName inductive_base_01 --save_path ./models/237_ind_rel_dub > 237_ind_rel_dub.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_sampler.py --train --configName 237_base_02 --save_path ./models/237_rel_dub > 237_ind_dub.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_sampler.py --train --configName inductive_base_01 --save_path ./models/237_ind_entity_dub > 237_ind_rel_dub.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python hyper_graph_sampler.py --train --configName 237_base_02 --save_path ./models/237_entity_dub > 237_entity_dub.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python hyper_graph_sampler.py --train --configName wn18rr_base_01 --save_path ./models/wn18rr_base_01 > wn18rr_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python hyper_graph_sampler.py --train --configName wn18rr_base_02 --save_path ./models/wn18rr_base_02 > wn18rr_base_01.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_sampler.py --train --configName inductive_base_01 --save_path ./models/237_ind_debug > 237_ind_debug.log 2>&1 &

# 使用余弦相似度进行计算
CUDA_VISIBLE_DEVICES=2 python hyper_graph_ce.py --train --configName base_01 --save_path ./models/ce_data_debug_01 
# 使用原始的得分计算方法
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName base_02 --save_path ./models/ce_data_debug_02 &

# 加入了shared relation 使用原始的score
CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_03 --save_path ./models/ce_data_debug_03 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_04 --save_path ./models/ce_data_debug_04 &

# 加入了shared relation 使用余弦计算
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce.py --train --configName base_05 --save_path ./models/ce_data_debug_05 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName base_06 --save_path ./models/ce_data_debug_06 &


CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_03 --save_path ./models/ce_data_debug_031 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_04 --save_path ./models/ce_data_debug_041 &


CUDA_VISIBLE_DEVICES=6 python hyper_graph_ce.py --train --configName base_07 --save_path ./models/ce_data_debug_07

CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce.py --train --configName base_08 --save_path ./models/ce_data_debug_08 &



CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_entity.py --train --configName wn18rr_base_01 --save_path ./models/wn18rr_hyper_base_01 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_entity.py --train --configName wn18rr_base_02 --save_path ./models/wn18rr_hyper_base_02 &



# CUDA_VISIBLE_DEVICES=4 python hyper_graph_sampler.py --train --configName inductive_base_01 --save_path ./models/237_ind_link_prediction 

CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_03 --save_path ./models/ce_data_new_03 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_04 --save_path ./models/ce_data_new_04 &

# 直接预测关系的lable
CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_01 --save_path ./models/ce_data_new_01 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_02 --save_path ./models/ce_data_new_02 &

# 添加了酶的属性
CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_03 --save_path ./models/ce_data_new_03 &

CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_04 --save_path ./models/ce_data_new_04 &

0c1cb70
CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_031 --save_path ./models/ce_data_rel_cat_01 &

af77b4293e013966b44ce3fd5a67f7e41db5da48
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_031 --save_path ./models/ce_data_score_cat_01 &

b5e39ba
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_031 --save_path ./models/ce_data_rel_add_score_cat_01 &
5e25a3e
CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_031 --save_path ./models/ce_data_rel_add_score_mul_pre_01 &


CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce.py --train --configName base_032 --save_path ./models/ce_data_rel_cat_score_mul_pre_01 &

CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName base_033 --save_path ./models/ce_data_rel_add_con_loss_score_mul_pre_01 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce.py --train --configName base_034 --save_path ./models/ce_data_rel_add_con_loss_score_mul_pre_02 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce.py --train --configName base_035 --save_path ./models/ce_data_rel_add_con_loss_score_mul_pre_03 &



CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce.py --train --configName base_032 --save_path ./models/ce_data_rel_add_score_mul_01 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce.py --train --configName base_032 --save_path ./models/ce_data_rel_add_score_combine_01 &

CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce.py --train --configName base_032 --save_path ./models/ce_data_rel_add_score_combine_01 &

CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName base_032 --save_path ./models/ce_data_debug_01 &

CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_0202_01 --save_path ./models/ce_data_debug_02 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce.py --train --configName base_0202_02 --save_path ./models/ce_data_debug_03 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName base_0202_03 --save_path ./models/ce_data_debug_04 &





CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce.py --train --configName base_032 --save_path     ./models/ce_data_type_emb_01 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce.py --train --configName base_0202_01 --save_path ./models/ce_data_type_emb_02 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce.py --train --configName base_0202_02 --save_path ./models/ce_data_type_emb_03 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce.py --train --configName base_0202_03 --save_path ./models/ce_data_type_emb_04 &