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

# cat init emb
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce.py --train --configName cat_0203_02 --save_path ./models/ce_data_type_emb_05 &

CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce_v2.py --train --configName base_032 --save_path     ./models/ce_data_v2_01 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_01 --save_path ./models/ce_data_v2_02 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_02 --save_path ./models/ce_data_v2_03 &
# 共享参数
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_01 --save_path ./models/ce_data_v2_04 &

CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_03 --save_path ./models/ce_data_v2_05 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_04 --save_path ./models/ce_data_v2_06 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_05 --save_path ./models/ce_data_v2_07 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_06 --save_path ./models/ce_data_v2_08 &
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_07 --save_path ./models/ce_data_v2_09 &


CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce.py --train --configName base_032 --save_path ./models/v1_debug &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName base_0202_01 --save_path ./models/v1_debug_01 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce.py --train --configName base_0202_02 --save_path ./models/v1_debug_02 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName base_0202_03 --save_path ./models/v1_debug_03 &


CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce.py --train --configName stw_0206 --save_path ./models/v1_debug &



CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce_v2.py --train --configName yinlong_0213_01 --save_path ./models/ce_data_v2_parameters_01 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v2.py --train --configName yinlong_0213_02 --save_path ./models/ce_data_v2_parameters_02 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v2.py --train --configName yinlong_0213_03 --save_path ./models/ce_data_v2_parameters_03 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName yinlong_0213_04 --save_path ./models/ce_data_v2_parameters_04 &


CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName yinlong_0213_01 --save_path ./models/ce_data_v1_parameters_01 --init  &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName yinlong_0213_02 --save_path ./models/ce_data_v1_parameters_02 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce.py --train --configName yinlong_0213_03 --save_path ./models/ce_data_v1_parameters_03 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName yinlong_0213_04 --save_path ./models/ce_data_v1_parameters_04 &



CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName yinlong_0213_01 --save_path ./models/ce_data_v1_old_parameters_01 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v1_old_parameters_01 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName yinlong_0213_02 --save_path ./models/ce_data_v1_old_parameters_02 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce.py --train --configName yinlong_0213_03 --save_path ./models/ce_data_v1_old_parameters_03 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName yinlong_0213_04 --save_path ./models/ce_data_v1_old_parameters_04 &
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce.py --train --configName yinlong_0213_05 --save_path ./models/ce_data_v1_old_parameters_05 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce.py --train --configName yinlong_0213_06 --save_path ./models/ce_data_v1_old_parameters_06 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce.py --train --configName yinlong_0213_07 --save_path ./models/ce_data_v1_old_parameters_07 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce.py --train --configName yinlong_0213_08 --save_path ./models/ce_data_v1_old_parameters_08 &



CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_03 --save_path ./models/ce_data_v2_add_mlp1_01 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_04 --save_path ./models/ce_data_v2_add_mlp1_02 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_05 --save_path ./models/ce_data_v2_add_mlp1_03 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_06 --save_path ./models/ce_data_v2_add_mlp1_04 &



CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_03 --save_path ./models/ce_data_v2_add_mlp2_01 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_04 --save_path ./models/ce_data_v2_add_mlp2_02 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_05 --save_path ./models/ce_data_v2_add_mlp2_03 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v2.py --train --configName cl_0203_06 --save_path ./models/ce_data_v2_add_mlp2_04 &


CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName yinlong_0213_01 --save_path ./models/ce_data_v1_old_parameters_01_test --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v1_old_parameters_01 &


CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName yinlong_0220_01 --save_path ./models/ce_data_v1_old_parameters_11 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName yinlong_0220_02 --save_path ./models/ce_data_v1_old_parameters_12 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce.py --train --configName yinlong_0220_04 --save_path ./models/ce_data_v1_old_parameters_14 &

## add test batch_size 和 采样的size
50,10
CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --train --configName yinlong_0220_03 --save_path ./models/ce_data_v1_old_parameters_13_1 &
50,15
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce.py --train --configName yinlong_0220_03 --save_path ./models/ce_data_v1_old_parameters_13_2 &


CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_01 --save_path ./models/ce_data_v2_mlp2_change_nerbor_size_01 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_02 --save_path ./models/ce_data_v2_mlp2_change_nerbor_size_02 &


CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce.py --train --configName yinlong_0220_031 --save_path ./models/ce_data_v1_old_parameters_change_dropout_11 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce.py --train --configName yinlong_0220_032 --save_path ./models/ce_data_v1_old_parameters_change_dropout_12 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce.py --train --configName yinlong_0220_033 --save_path ./models/ce_data_v1_old_parameters_change_dropout_14 &
# CUDA_VISIBLE_DEVICES=8 nohup python hyper_graph_ce.py --train --configName yinlong_0220_034 --save_path ./models/ce_data_v1_old_parameters_change_dropout_14 &



CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce.py --configName yinlong_0213_01 --save_path ./models/ce_data_v1_old_parameters_01_test --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v1_old_parameters_01 &



CUDA_VISIBLE_DEVICES=0 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_01 --save_path ./models/ce_data_v2_0222_01 &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_02 --save_path ./models/ce_data_v2_0222_02 &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_03 --save_path ./models/ce_data_v2_0222_03 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_04 --save_path ./models/ce_data_v2_0222_04 &


CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_01 --save_path ./models/ce_data_v2_0222_sgd_01 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_02 --save_path ./models/ce_data_v2_0222_sgd_02 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_03 --save_path ./models/ce_data_v2_0222_sgd_03 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v2.py --train --configName cl_0220_04 --save_path ./models/ce_data_v2_0222_sgd_04 &


CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_01 --save_path ./models/ce_data_v2_0222_sgd_05 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_0222_sgd_06 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_0222_sgd_07 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_0222_sgd_08 &

# 
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_01 --save_path ./models/ce_data_v2_base_predict_0224_01 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_base_predict_0224_02 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_base_predict_0224_03 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_base_predict_0224_04 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_05 --save_path ./models/ce_data_v2_base_predict_0224_05 &


CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_01 --save_path ./models/ce_data_v2_base_predict_0224_11 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_base_predict_0224_12 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_base_predict_0224_13 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_base_predict_0224_14 &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v2.py --train --configName cl_0221_05 --save_path ./models/ce_data_v2_base_predict_0224_15 &

CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_01 --save_path ./models/ce_data_v2_baseline_0227_01 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_baseline_0227_02 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_baseline_0227_03 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_baseline_0227_04 &

CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_01 --save_path ./models/ce_data_v2_baseline_0227_11 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_baseline_0227_12 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_baseline_0227_13 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_baseline_0227_14 &

# 只用score 得分
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_01 --save_path ./models/ce_data_v2_baseline_0227_21 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_baseline_0227_22 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_baseline_0227_23 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_baseline_0227_24 &

# add 对比学习
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_01 --save_path ./models/ce_data_v2_baseline_0227_31 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_baseline_0227_32 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_baseline_0227_33 &
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_baseline_0227_34 &


CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v4.py --configName cl_0221_01 --save_path ./models/ce_data_v4_0227_31 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_31/hit10/ &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v4.py --configName cl_0221_02 --save_path ./models/ce_data_v4_0227_32 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_32/hit10/ &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v4.py --configName cl_0221_03 --save_path ./models/ce_data_v4_0227_33 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_33/hit10/ &
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v4.py --configName cl_0221_04 --save_path ./models/ce_data_v4_0227_34 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_34/hit10/ &


# single score
# CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v4.py --configName cl_0221_01 --save_path ./models/ce_data_v4_0227_21  --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_21/hit10/ &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v4.py --configName cl_0221_02 --save_path ./models/ce_data_v4_0227_22 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_22/hit10/ &
CUDA_VISIBLE_DEVICES=2 nohup python hyper_graph_ce_v4.py --configName cl_0221_03 --save_path ./models/ce_data_v4_0227_23 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_23/hit10/ &
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v4.py --configName cl_0221_04 --save_path ./models/ce_data_v4_0227_24 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v4_0227_24/hit10/ &



CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v3.py --configName cl_0221_01 --save_path ./models/ce_data_v2_baseline_0227_31 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_31/hit10/ &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v3.py --configName cl_0221_02 --save_path ./models/ce_data_v2_baseline_0227_32 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_32/hit10/ &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v3.py --configName cl_0221_03 --save_path ./models/ce_data_v2_baseline_0227_33 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_33/hit10/ &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v3.py --configName cl_0221_04 --save_path ./models/ce_data_v2_baseline_0227_34 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_34/hit10/ &


CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v3.py --configName cl_0221_01 --save_path ./models/ce_data_v2_baseline_0227_11 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_11/hit1/ &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v3.py --configName cl_0221_02 --save_path ./models/ce_data_v2_baseline_0227_12 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_12/hit1/ &
CUDA_VISIBLE_DEVICES=1 nohup python hyper_graph_ce_v3.py --configName cl_0221_03 --save_path ./models/ce_data_v2_baseline_0227_13 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_13/hit1/ &
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v3.py --configName cl_0221_04 --save_path ./models/ce_data_v2_baseline_0227_14 --init /home/skl/yl/ce_project/relation_cl/models/models/ce_data_v2_baseline_0227_14/hit1/ &




CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_best --save_path ./models/ce_data_v3_debug &

CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v3.py --train --configName cl_03_01 --save_path ./models/ce_data_v3_0303_1 &
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v3.py --train --configName cl_03_02 --save_path ./models/ce_data_v3_0303_2 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v3.py --train --configName cl_03_03 --save_path ./models/ce_data_v3_0303_3 &

# CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v3.py --train --configName cl_03_01 --save_path ./models/ce_data_v3_0303_1 &
# add mlp
CUDA_VISIBLE_DEVICES=3 nohup python hyper_graph_ce_v3.py --train --configName cl_03_02 --save_path ./models/ce_data_v3_0304_1 &
CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v3.py --train --configName cl_03_03 --save_path ./models/ce_data_v3_0304_2 &
# only cl learning
CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v3.py --train --configName cl_03_03 --save_path ./models/ce_data_v3_0304_3 &

CUDA_VISIBLE_DEVICES=5 python hyper_graph_ce_v3.py --train --configName cl_0221_best --save_path ./models/ce_data_v3_add_help_0304_1 &






# CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_02 --save_path ./models/ce_data_v2_baseline_0227_32 &
# CUDA_VISIBLE_DEVICES=6 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_03 --save_path ./models/ce_data_v2_baseline_0227_33 &
# CUDA_VISIBLE_DEVICES=7 nohup python hyper_graph_ce_v3.py --train --configName cl_0221_04 --save_path ./models/ce_data_v2_baseline_0227_34 &


# 去掉对比学习, 混合score
CUDA_VISIBLE_DEVICES=4 nohup python hyper_graph_ce_v4.py --train --configName cl_0221_02 --save_path ./models/ce_data_v4_no_cl_0304_1 &
# 去掉对比学习， 混合score 同一个embedding混合
CUDA_VISIBLE_DEVICES=5 nohup python hyper_graph_ce_v4.py --train --configName cl_0221_02 --save_path ./models/ce_data_v4_no_cl_mix_score_0304_1 &
